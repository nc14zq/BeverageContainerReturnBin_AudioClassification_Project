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
#include "/Users/nc/NTU/BCR_AudioProject/headerfiles/fft_bcr_float32.h"

///////////////////////////////////////////////////////// AUDIO SETUP
constexpr uint32_t SAMPLE_RATE = 16000;
constexpr uint16_t FFT_SIZE    = 4096;
constexpr uint8_t  CHANNELS    = 1;

constexpr float PEAK_THRESHOLD = 0.15f;

///////////////////////////////////////////////////////// IMU SETUP
constexpr float IMU_THRESHOLD_G = 1.05f;
constexpr unsigned long IMU_POLL_MS = 25;          // low-rate IMU polling
constexpr unsigned long REARM_COOLDOWN_MS = 250;   // short cooldown after event

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

// FFT buffers
double vReal[FFT_SIZE];
double vImag[FFT_SIZE];

ArduinoFFT<double> FFT = ArduinoFFT<double>(vReal, vImag, FFT_SIZE, SAMPLE_RATE);

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
constexpr int kTensorArenaSize = 12 * 1024;
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

///////////////////////////////////////// SPECTRAL CENTROID

float computeSpectralCentroid(const double* mags, int bins, float binHz) {
  double weightedSum = 0.0;
  double magSum = 0.0;

  for (int i = 1; i < bins; i++) {
    double freq = i * binHz;
    weightedSum += freq * mags[i];
    magSum += mags[i];
  }

  if (magSum < 1e-12) return 0.0f;
  return (float)(weightedSum / magSum);
}

///////////////////////////////////////// SPECTRAL FLATNESS

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

///////////////////////////////////////// BAND ENERGY RATIO

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

  if (totalSum < 1e-12) return 0.0f;
  return (float)(bandSum / totalSum);
}

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

void scaleFeatures(const float raw_features[4], float scaled_features[4]) {
  for (int i = 0; i < 4; i++) {
    scaled_features[i] = (raw_features[i] - mean_vals[i]) / scale_vals[i];
  }
}

bool runInference(const float scaled_features[4], float& prob) {
  if (input->type == kTfLiteFloat32) {
    for (int i = 0; i < 4; i++) {
      input->data.f[i] = scaled_features[i];
    }
  } 
  
  else {
    return false;
  }

  TfLiteStatus status = interpreter->Invoke();

  if (status != kTfLiteOk) {
    Serial.println("Invoke failed");
    return false;
  }

  if (output->type == kTfLiteFloat32) {
    prob = output->data.f[0];
  } else {
    Serial.println("Unsupported output type");
    return false;
  }

  return true;
}

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
}

///////////////////////////////////////////////////////// PDM CALLBACK

void onPDMdata() {
  int bytesAvailable = PDM.available();
  if (bytesAvailable > (int)sizeof(pdmBuffer)) {
    bytesAvailable = sizeof(pdmBuffer);
  }

  PDM.read(pdmBuffer, bytesAvailable);

  if (!captureAudio) return;

  int sampleCount = bytesAvailable / 2;

  for (int i = 0; i < sampleCount; i++) {
    if (audioIndex < FFT_SIZE) {
      audioFrame[audioIndex++] = (float)pdmBuffer[i] / 32768.0f;
    }

    if (audioIndex >= FFT_SIZE) {
      frameReady = true;
      captureAudio = false;
      break;
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

    noInterrupts();
    audioIndex = 0;
    frameReady = false;
    captureAudio = true;
    interrupts();
  }
}

///////////////////////////////////////////////////////// SERVO ACTION

unsigned long moveServoForClass(int predictedClass) {
  unsigned long sortStartUs = micros();

  sorterServo.attach(SERVO_PIN);
  delay(SERVO_ATTACH_SETTLE_MS);

  if (predictedClass == 1) {
    sorterServo.write(SERVO_RIGHT_ANGLE);
  } else {
    sorterServo.write(SERVO_LEFT_ANGLE);
  }

  delay(SERVO_HOLD_MS);

  sorterServo.write(SERVO_CENTER_ANGLE);

  delay(SERVO_RETURN_MS);

  sorterServo.detach();   // servo disarmed when idle

  unsigned long sortElapsedUs = micros() - sortStartUs;

  rearmSystem();
  return sortElapsedUs;
}

///////////////////////////////////////////////////////// SETUP

void setup() {
  Serial.begin(115200);
  while (!Serial) {}

  if (!IMU.begin()) {
    Serial.println("IMU init failed");
    while (1);
  }

  sorterServo.attach(SERVO_PIN);
  delay(SERVO_ATTACH_SETTLE_MS);
  sorterServo.write(SERVO_CENTER_ANGLE);
  delay(500);
  sorterServo.detach();

  model = tflite::GetModel(fft_bcr_float32_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch");
    while (1);
  }

  static tflite::MicroMutableOpResolver<2> resolver;
  resolver.AddFullyConnected(); //DENSE LAYER
  resolver.AddLogistic(); //SIGMOID OUTPUT

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, nullptr, nullptr);

  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed");
    while (1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  PDM.onReceive(onPDMdata);
  PDM.setGain(30);

  if (!PDM.begin(CHANNELS, SAMPLE_RATE)) {
    Serial.println("PDM.begin() failed");
    while (1);
  }

  resetAudioFrame();
}

// =====================
// LOOP
// =====================
void loop() {
  checkIMUTriggerLowRate();

  if (!frameReady || !imuTriggered) return;

  noInterrupts();
  frameReady = false;
  interrupts();

  float rawPeak = computePeak(audioFrame, FFT_SIZE);
  
  if (rawPeak < PEAK_THRESHOLD) {
    rearmSystem();
    return;
  }

  float raw_features[4];
  float scaled_features[4];
  float prob = 0.0f;

  extractFeatures(raw_features);
  scaleFeatures(raw_features, scaled_features);

  if (runInference(scaled_features, prob)) {

    int predictedClass = (prob >= 0.5f) ? 1 : 0;

    if (predictedClass == 1) {moveServoForClass(1);} 
    else {moveServoForClass(0);}

  } else {rearmSystem();}

  
}