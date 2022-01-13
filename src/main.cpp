
#include <iostream>
#include <string>
#include <fstream>
#include "secondary.h"
#include "connect.h"
#include "AESObject.h"
#include "NeuralNetConfig.h"
#include "NeuralNetwork.h"
#include "Functionalities.h"
#include <vector>
#include "tools.h"

using namespace std;
extern int partyNum;
int NUM_OF_PARTIES;

AESObject *aes_common;
AESObject *aes_indep;
AESObject *aes_a_1;
AESObject *aes_a_2;
AESObject *aes_b_1;
AESObject *aes_b_2;
AESObject *aes_c_1;
ParallelAESObject *aes_parallel;

int main(int argc, char **argv)
{

  /****************************** PREPROCESSING ******************************/
  parseInputs(argc, argv);

  // for (size_t i = 0; i < argc; i++)
  //   cout << "这是第" << i << "个参数" << argv[i] << endl;

  string whichNetwork = "No Network";
  NeuralNetConfig *config = new NeuralNetConfig(NUM_ITERATIONS);
  /****************************** SELECT NETWORK ******************************/
  // MINIONN, Network - D in GAZELLE

  // whichNetwork = "MiniONN/GAZELLE-D";
  // CNNConfig *l0 = new CNNConfig(16, 1, 5, 5, MINI_BATCH_SIZE, 28, 28, 2, 2);
  // CNNConfig *l1 = new CNNConfig(16, 16, 5, 5, MINI_BATCH_SIZE, 12, 12, 2, 2);
  // FCConfig *l2 = new FCConfig(MINI_BATCH_SIZE, 256, 100);
  // FCConfig *l3 = new FCConfig(MINI_BATCH_SIZE, 100, 10);
  // config->addLayer(l0);
  // config->addLayer(l1);
  // config->addLayer(l2);
  // config->addLayer(l3);

  //LeNet
  whichNetwork = "LeNet";
  CNNConfig *l0 = new CNNConfig(20, 1, 5, 5, MINI_BATCH_SIZE, 28, 28, 2, 2);
  CNNConfig *l1 = new CNNConfig(50, 20, 5, 5, MINI_BATCH_SIZE, 12, 12, 2, 2);
  FCConfig *l2 = new FCConfig(MINI_BATCH_SIZE, 800, 500);
  FCConfig *l3 = new FCConfig(MINI_BATCH_SIZE, 500, 10);
  config->addLayer(l0);
  config->addLayer(l1);
  config->addLayer(l2);
  config->addLayer(l3);

  //SecureML
  // whichNetwork = "SecureML";
  // FCConfig *l0 = new FCConfig(MINI_BATCH_SIZE, LAYER0, LAYER1);
  // FCConfig *l1 = new FCConfig(MINI_BATCH_SIZE, LAYER1, LAYER2);
  // FCConfig *l2 = new FCConfig(MINI_BATCH_SIZE, LAYER2, LAST_LAYER_SIZE);
  // config->addLayer(l0);
  // config->addLayer(l1);
  // config->addLayer(l2);

  //Chameleon
  // whichNetwork = "Sarda";
  // ChameleonCNNConfig *l0 = new ChameleonCNNConfig(5, 1, 5, 5, MINI_BATCH_SIZE, 28, 28, 2, 2);
  // FCConfig *l1 = new FCConfig(MINI_BATCH_SIZE, 980, 100);
  // FCConfig *l2 = new FCConfig(MINI_BATCH_SIZE, 100, 10);
  // config->addLayer(l0);
  // config->addLayer(l1);
  // config->addLayer(l2);

  config->checkNetwork();
  NeuralNetwork *network = new NeuralNetwork(config);

  /****************************** AES SETUP and SYNC ******************************/
  aes_indep = new AESObject(argv[4]);
  aes_common = new AESObject(argv[5]);
  aes_a_1 = new AESObject("files/keyD");
  aes_a_2 = new AESObject("files/keyD");
  aes_b_1 = new AESObject("files/keyD");
  aes_b_2 = new AESObject("files/keyD");
  aes_c_1 = new AESObject("files/keyD");
  aes_parallel = new ParallelAESObject(argv[5]);

  if (!STANDALONE)
  {
    initializeCommunication(argv[3], partyNum);
    // initializeMPC();
    synchronize(2000000);
  }

  if (PARALLEL)
    aes_parallel->precompute();
  // cout << 'start';
  // fstream f;
  // fstream fs;
  // fs.open("data.txt", ios::in);
  // if (!fs)
  // {
  //   ofstream fout("data.txt");
  // }
  // fs.close();
  // f.open("data.txt", ios::out | ios::app);
  // f << "开始写入" << '\n';
  // for (int i = 0; i < 256; i++)
  // {
  //   f << (int)aes_a_1->get8Bits() << ' ';
  // }

  // f << '\n'
  //   << "写入结束" << '\n';
  // f.close();
  // cout << endl;

  // int size1 = 100;
  // vector<myType> A1(size1, 0);
  // populateRandomVector<myType>(A1, size1, "COMMON", "NEGATIVE");
  // fstream f;
  // f.open("negative.txt", ios::out);
  // for (int i = 0; i < 100; i++)
  //   f << (unsigned long)A1[i] << ' ';
  // f.close();

  /****************************** RUN NETWORK/BENCHMARKS ******************************/
  start_m();

  // whichNetwork = "Mat-Mul";
  // testMatMul(784, 128, 10, NUM_ITERATIONS);
  // testMatMul(1, 500, 100, NUM_ITERATIONS);
  // testMatMul(1, 100, 1, NUM_ITERATIONS);

  // whichNetwork = "Convolution";
  // testConvolution(28, 28, 5, 5, 1, 20, NUM_ITERATIONS);
  // testConvolution(28, 28, 3, 3, 1, 20, NUM_ITERATIONS);
  // testConvolution(8, 8, 5, 5, 16, 50, NUM_ITERATIONS);

  // whichNetwork = "Relu";
  // testRelu(128, 128, NUM_ITERATIONS);
  // testRelu(576, 20, NUM_ITERATIONS);
  // testRelu(64, 16, NUM_ITERATIONS);

  // whichNetwork = "ReluPrime";
  // testReluPrime(128, 128, NUM_ITERATIONS);
  // testReluPrime(576, 20, NUM_ITERATIONS);
  // testReluPrime(64, 16, NUM_ITERATIONS);

  // whichNetwork = "MaxPool";
  // testMaxPool(24, 24, 2, 2, 20, NUM_ITERATIONS);
  // testMaxPool(24, 24, 2, 2, 16, NUM_ITERATIONS);
  // testMaxPool(8, 8, 4, 4, 50, NUM_ITERATIONS);

  // whichNetwork = "MaxPoolDerivative";
  // testMaxPoolDerivative(24, 24, 2, 2, 20, NUM_ITERATIONS);
  // testMaxPoolDerivative(24, 24, 2, 2, 16, NUM_ITERATIONS);
  // testMaxPoolDerivative(8, 8, 4, 4, 50, NUM_ITERATIONS);

  whichNetwork += " train";
  train(network, config);
  // vector<myType> t1(110, 0);
  // for (size_t i = 0; i < 110; i++)
  //   t1[i] = i;
  // printMatShape("a", t1, 11, 11, false);

  whichNetwork += " test";
  test(network);
  // debugDivision();
  // debugMatMul(); //it's correct
  // debugComputeMSB(); //wrong
  // debugComputeShareConvert(); //wrong
  // debugMaxIndex();
  // debugRELU();
  // debugPC(); //wrong
  // debugMax(); //wrong
  // debugSS(); //select shares it's correct
  // debugReLUPrime(); //wrong

  end_m(whichNetwork);
  cout << "----------------------------------" << endl;
  cout << NUM_OF_PARTIES << "PC code, P" << partyNum << endl;
  cout << NUM_ITERATIONS << " iterations, " << whichNetwork << ", batch size " << MINI_BATCH_SIZE << endl;
  cout << "----------------------------------" << endl
       << endl;

  /****************************** CLEAN-UP ******************************/
  delete aes_common;
  delete aes_indep;
  delete aes_a_1;
  delete aes_a_2;
  delete aes_b_1;
  delete aes_b_2;
  delete aes_c_1;
  delete aes_parallel;
  delete config;
  delete l0;
  delete l1;
  delete l2;
  // delete l3;
  delete network;
  if (partyNum != PARTY_S)
    deleteObjects();

  return 0;
}
