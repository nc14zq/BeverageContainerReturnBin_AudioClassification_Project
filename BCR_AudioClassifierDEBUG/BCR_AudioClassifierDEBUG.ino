#include <Arduino.h>
#include <PDM.h>
#include <math.h>
#include <Servo.h>

#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "arduinoFFT.h"
#include <Arduino_LSM9DS1.h>
#include "fft_bcr_float32.h"

///////////////////////////////////////////////////////// AUDIO SETUP

constexpr uint32_t SAMPLE_RATE = 16000;
constexpr uint16_t FFT_SIZE    = 4096;
constexpr uint8_t  CHANNELS    = 1;

constexpr float PEAK_THRESHOLD = 0.15f;

constexpr uint16_t PRE_TRIGGER_SAMPLES = 2400;

///////////////////////////////////////////////////////// IMU SETUP

constexpr float IMU_THRESHOLD_G = 1.05f;
constexpr unsigned long IMU_POLL_MS = 25; //POWER-EFFICIENCY OPTIMISATION          
constexpr unsigned long REARM_COOLDOWN_MS = 250; //POWER-EFFICIENCY OPTIMISATION

///////////////////////////////////////////////////////// SERVO SETUP

constexpr int SERVO_PIN = 9;
constexpr int SERVO_CENTER_ANGLE = 90;
constexpr int SERVO_LEFT_ANGLE   = 0;
constexpr int SERVO_RIGHT_ANGLE  = 180;

constexpr unsigned long SERVO_ATTACH_SETTLE_MS = 120;
constexpr unsigned long SERVO_HOLD_MS   = 700;
constexpr unsigned long SERVO_RETURN_MS = 700;

Servo sorterServo;

///////////////////////////////////////////////////////// SYSTEM STATE

volatile bool imuTriggered = false;
bool systemArmed = true;

unsigned long lastImuPollMs = 0;
unsigned long rearmBlockedUntilMs = 0;

// PDM callback buffer
int16_t pdmBuffer[512];

// captured event audio
float audioFrame[FFT_SIZE];
volatile uint16_t audioIndex = 0;
volatile bool frameReady = false;
volatile bool captureAudio = false;

// pre-trigger rolling buffer
float preTriggerBuffer[PRE_TRIGGER_SAMPLES];
volatile uint16_t preTriggerWriteIndex = 0;
volatile bool preTriggerFilled = false;

// FFT buffers
double vReal[FFT_SIZE];
double vImag[FFT_SIZE];

ArduinoFFT<double> FFT = ArduinoFFT<double>(vReal, vImag, FFT_SIZE, SAMPLE_RATE);

///////////////////////////////////////////////////////// LATENCY TRACKING

unsigned long eventStartUs = 0;          
unsigned long featureLatencyUs = 0;      
unsigned long inferenceLatencyUs = 0;    
unsigned long sortingLatencyUs = 0;      
unsigned long totalLatencyUs = 0;        

///////////////////////////////////////////////////////// SCALER SETUP

const float mean_vals[4] = {
  1653.32744f,
  0.322853950f,
  0.184309233f,
  0.138988863f,
};

const float scale_vals[4] = {
  528.569917f,
  0.108196103f,
  0.171209716f,
  0.198178077f,
};

///////////////////////////////////////////////////////// TFLM

constexpr int kTensorArenaSize = 3 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

///////////////////////////////////////////////////////// HELPERS

float computePeak(const float* buf, int n) {
  float peak = 0.0f;
  for (int i = 0; i < n; i++) {
    float v = fabs(buf[i]);
    if (v > peak) peak = v;
  }
  return peak;
}

void normalizeAudio(float* buf, int n) {
  float peak = computePeak(buf, n);
  if (peak <= 1e-12f) return;

  for (int i = 0; i < n; i++) {
    buf[i] /= peak;
  }
}

void resetPreTriggerBuffer() {
  noInterrupts();
  preTriggerWriteIndex = 0;
  preTriggerFilled = false;
  interrupts();

  for (int i = 0; i < PRE_TRIGGER_SAMPLES; i++) {
    preTriggerBuffer[i] = 0.0f;
  }
}

void copyPreTriggerToAudioFrame() {
  uint16_t startIdx = preTriggerFilled ? preTriggerWriteIndex : 0;
  uint16_t available = preTriggerFilled ? PRE_TRIGGER_SAMPLES : preTriggerWriteIndex;
  uint16_t padCount = PRE_TRIGGER_SAMPLES - available;
  uint16_t outIdx = 0;

  for (uint16_t i = 0; i < padCount; i++) {
    audioFrame[outIdx++] = 0.0f;
  }

  if (preTriggerFilled) {
    for (uint16_t i = 0; i < PRE_TRIGGER_SAMPLES; i++) {
      uint16_t idx = (startIdx + i) % PRE_TRIGGER_SAMPLES;
      audioFrame[outIdx++] = preTriggerBuffer[idx];
    }
  } else {
    for (uint16_t i = 0; i < available; i++) {
      audioFrame[outIdx++] = preTriggerBuffer[i];
    }
  }
}

///////////////////////////////// SPECTRAL CENTROID

float computeSpectralCentroid(const double* mags, int bins, float binHz) {
  double weightedSum = 0.0;
  double magSum = 0.0;

  for (int i = 1; i < bins; i++) {
    double freq = i * binHz;
    weightedSum += freq * mags[i];
    magSum += mags[i];
  }

  if (magSum < 1e-12f) return 0.0f;
  return (float)(weightedSum / magSum);
}

///////////////////////////////// SPECTRAL FLATNESS

float computeSpectralFlatness(const double* mags, int bins) {
  const double eps = 1e-12;
  double logSum = 0.0;
  double arithSum = 0.0;
  int count = 0;

  for (int i = 1; i < bins; i++) {
    double m = mags[i] + eps;
    logSum += log(m);
    arithSum += m;
    count++;
  }

  if (count == 0 || arithSum < eps) return 0.0f;

  double geoMean = exp(logSum / count);
  double arithMean = arithSum / count;
  return (float)(geoMean / arithMean);
}

///////////////////////////////// BAND ENERGY RATIO

float bandEnergyRatioPower(const double* mags, int bins, float binHz, float fLow, float fHigh) {
  double bandSum = 0.0;
  double totalSum = 0.0;

  for (int i = 1; i < bins; i++) {
    double power = mags[i] * mags[i];
    float f = i * binHz;
    totalSum += power;
    if (f >= fLow && f < fHigh) {
      bandSum += power;
    }
  }

  if (totalSum < 1e-12f) return 0.0f;
  return (float)(bandSum / totalSum);
}

///////////////////////////////// EXTRACT FEATURES

void extractFeatures(float raw_features[4]) {
  normalizeAudio(audioFrame, FFT_SIZE);

  for (int i = 0; i < FFT_SIZE; i++) {
    double w = 0.5 * (1.0 - cos((2.0 * PI * i) / (FFT_SIZE - 1)));
    vReal[i] = (double)audioFrame[i] * w;
    vImag[i] = 0.0;
  }

  FFT.compute(FFT_FORWARD);
  FFT.complexToMagnitude();

  const int bins = FFT_SIZE / 2;
  const float binHz = (float)SAMPLE_RATE / (float)FFT_SIZE;

  raw_features[0] = computeSpectralCentroid(vReal, bins, binHz);
  raw_features[1] = computeSpectralFlatness(vReal, bins);
  raw_features[2] = bandEnergyRatioPower(vReal, bins, binHz, 1000.0f, 2000.0f);
  raw_features[3] = bandEnergyRatioPower(vReal, bins, binHz, 0.0f, 500.0f);
}

///////////////////////////////// SCALE FEATURES

void scaleFeatures(const float raw_features[4], float scaled_features[4]) {
  for (int i = 0; i < 4; i++) {
    scaled_features[i] = (raw_features[i] - mean_vals[i]) / scale_vals[i];
  }
}

///////////////////////////////// RUN INFERENCE

bool runInference(const float scaled_features[4], float& prob) {
  if (input->type == kTfLiteFloat32) {
    for (int i = 0; i < 4; i++) {
      input->data.f[i] = scaled_features[i];
    }
  } else if (input->type == kTfLiteInt8) {
    float in_scale = input->params.scale;
    int in_zero = input->params.zero_point;

    for (int i = 0; i < 4; i++) {
      int32_t q = (int32_t)round(scaled_features[i] / in_scale + in_zero);
      if (q > 127) q = 127;
      if (q < -128) q = -128;
      input->data.int8[i] = (int8_t)q;
    }
  } else {
    Serial.println("Unsupported input type");
    return false;
  }

  unsigned long inferStartUs = micros();
  TfLiteStatus status = interpreter->Invoke();
  inferenceLatencyUs = micros() - inferStartUs;

  if (status != kTfLiteOk) {
    Serial.println("Invoke failed");
    return false;
  }

  if (output->type == kTfLiteFloat32) {
    prob = output->data.f[0];
  } else if (output->type == kTfLiteInt8) {
    prob = (output->data.int8[0] - output->params.zero_point) * output->params.scale;
  } else {
    Serial.println("Unsupported output type");
    return false;
  }

  return true;
}

///////////////////////////////// SENSORS RESET

void resetAudioFrame() {
  noInterrupts();
  audioIndex = 0;
  frameReady = false;
  captureAudio = false;
  interrupts();
}

void rearmSystem() {
  imuTriggered = false;
  systemArmed = true;
  rearmBlockedUntilMs = millis() + REARM_COOLDOWN_MS;
  resetAudioFrame();
  Serial.println("System re-armed");
}

///////////////////////////////////////////////////////// LATENCY TRACKING

void printLatencyBreakdown() {
  Serial.println("===== LATENCY BREAKDOWN =====");

  Serial.print("Feature extraction latency (ms): ");
  Serial.println(featureLatencyUs / 1000.0f, 3);

  Serial.print("Inference latency (ms): ");
  Serial.println(inferenceLatencyUs / 1000.0f, 3);

  Serial.print("Sorting latency (ms): ");
  Serial.println(sortingLatencyUs / 1000.0f, 3);

  Serial.print("Total latency (IMU trigger -> sorting done) (ms): ");
  Serial.println(totalLatencyUs / 1000.0f, 3);

  Serial.println("=============================");
}

///////////////////////////////////////////////////////// PDM CALLBACK

void onPDMdata() {
  int bytesAvailable = PDM.available();
  if (bytesAvailable > (int)sizeof(pdmBuffer)) {
    bytesAvailable = sizeof(pdmBuffer);
  }

  PDM.read(pdmBuffer, bytesAvailable);

  int sampleCount = bytesAvailable / 2;

  for (int i = 0; i < sampleCount; i++) {
    float sample = (float)pdmBuffer[i] / 32768.0f;

    preTriggerBuffer[preTriggerWriteIndex] = sample;
    preTriggerWriteIndex++;

    if (preTriggerWriteIndex >= PRE_TRIGGER_SAMPLES) {
      preTriggerWriteIndex = 0;
      preTriggerFilled = true;
    }

    if (captureAudio) {
      if (audioIndex < FFT_SIZE) {
        audioFrame[audioIndex++] = sample;
      }

      if (audioIndex >= FFT_SIZE) {
        frameReady = true;
        captureAudio = false;
        break;
      }
    }
  }
}

///////////////////////////////////////////////////////// IMU TRIGGER THRESHOLD

void checkIMUTriggerLowRate() {
  unsigned long now = millis();

  if (!systemArmed) return;
  if (captureAudio) return;
  if (now < rearmBlockedUntilMs) return;
  if (now - lastImuPollMs < IMU_POLL_MS) return;

  lastImuPollMs = now;

  if (!IMU.accelerationAvailable()) return;

  float x, y, z;
  IMU.readAcceleration(x, y, z);

  float mag = sqrt(x * x + y * y + z * z);

  if (mag > IMU_THRESHOLD_G) {
    imuTriggered = true;
    systemArmed = false;
    eventStartUs = micros();

    noInterrupts();
    frameReady = false;
    copyPreTriggerToAudioFrame();
    audioIndex = PRE_TRIGGER_SAMPLES;
    captureAudio = true;
    interrupts();

    Serial.print("IMU triggered, mag = ");
    Serial.println(mag, 3);
  }
}

///////////////////////////////////////////////////////// SERVO ACTION

unsigned long moveServoForClass(int predictedClass) {
  unsigned long sortStartUs = micros();

  sorterServo.attach(SERVO_PIN);
  delay(SERVO_ATTACH_SETTLE_MS);

  if (predictedClass == 1) {
    Serial.println("Servo -> CAN");
    sorterServo.write(SERVO_RIGHT_ANGLE);
  } else {
    Serial.println("Servo -> BOTTLE");
    sorterServo.write(SERVO_LEFT_ANGLE);
  }

  delay(SERVO_HOLD_MS);

  Serial.println("Servo -> CENTER");
  sorterServo.write(SERVO_CENTER_ANGLE);

  delay(SERVO_RETURN_MS);

  sorterServo.detach(); //POWER-EFFICIENCY OPTIMISATION

  unsigned long sortElapsedUs = micros() - sortStartUs;

  rearmSystem();
  return sortElapsedUs;
}

///////////////////////////////////////////////////////// SETUP

void setup() {
  Serial.begin(115200);
  while (!Serial) {}

  Serial.println("Booting up...");

  if (!IMU.begin()) {
    Serial.println("IMU init failed");
    while (1);
  }

/////////////////////////////////////// SERVO CENTERING

  sorterServo.attach(SERVO_PIN);
  delay(SERVO_ATTACH_SETTLE_MS);
  sorterServo.write(SERVO_CENTER_ANGLE);
  delay(500);
  sorterServo.detach();

/////////////////////////////////////// 

  model = tflite::GetModel(fft_bcr_float32_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch");
    while (1);
  }

  static tflite::MicroMutableOpResolver<4> resolver;
  resolver.AddFullyConnected();
  resolver.AddLogistic();
  resolver.AddQuantize();
  resolver.AddDequantize();

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, nullptr, nullptr);

  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed");
    while (1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.print("Input type: ");
  Serial.println(input->type);
  Serial.print("Output type: ");
  Serial.println(output->type);

  PDM.onReceive(onPDMdata);
  PDM.setGain(30);

  if (!PDM.begin(CHANNELS, SAMPLE_RATE)) {
    Serial.println("PDM.begin() failed");
    while (1);
  }

  resetAudioFrame();
  resetPreTriggerBuffer();
  Serial.println("Ready.");
}

///////////////////////////////////////////////////////// LOOP

void loop() {
  checkIMUTriggerLowRate();

  if (!frameReady || !imuTriggered) return;

  noInterrupts();
  frameReady = false;
  interrupts();

  float rawPeak = computePeak(audioFrame, FFT_SIZE);
  Serial.print("Raw peak: ");
  Serial.println(rawPeak, 6);

  if (rawPeak < PEAK_THRESHOLD) {
    Serial.println("Rejected: weak impact");
    totalLatencyUs = micros() - eventStartUs;

    Serial.print("Total latency (rejected weak impact) (ms): ");
    Serial.println(totalLatencyUs / 1000.0f, 3);

    Serial.println("----------------------");
    rearmSystem();
    return;
  }

  float raw_features[4];
  float scaled_features[4];
  float prob = 0.0f;

  unsigned long featureStartUs = micros();
  extractFeatures(raw_features);
  scaleFeatures(raw_features, scaled_features);
  featureLatencyUs = micros() - featureStartUs;

  Serial.println("Raw features:");
  for (int i = 0; i < 4; i++) {
    Serial.print(raw_features[i], 6);
    Serial.print(i < 3 ? ", " : "\n");
  }

  Serial.println("Scaled features:");
  for (int i = 0; i < 4; i++) {
    Serial.print(scaled_features[i], 6);
    Serial.print(i < 3 ? ", " : "\n");
  }

  if (runInference(scaled_features, prob)) {
    Serial.print("Probability: ");
    Serial.println(prob, 6);

    int predictedClass = (prob >= 0.5f) ? 1 : 0;

    if (predictedClass == 1) {
      Serial.println("Predicted class: Can");
      sortingLatencyUs = moveServoForClass(1);
    } else {
      Serial.println("Predicted class: Bottle");
      sortingLatencyUs = moveServoForClass(0);
    }

    totalLatencyUs = micros() - eventStartUs;
    printLatencyBreakdown();
  } else {
    totalLatencyUs = micros() - eventStartUs;
    Serial.print("Inference failed. Total latency (ms): ");
    Serial.println(totalLatencyUs / 1000.0f, 3);
    rearmSystem();
  }

  Serial.println("----------------------");
}