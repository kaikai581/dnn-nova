/*
 * 
 * This file is to output the training and testing data for wrong sign studies
 * to hdf5 files.
 * 
 */

// Standard includes
#include <fstream>
#include <iostream>

// Boost includes
#include <boost/algorithm/string/predicate.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

// CVN includes
#include "CVN/art/CaffeNetHandler.h"
#include "CVN/func/TrainingData.h"

// ROOT includes
#include "TChain.h"
#include "TSystem.h"
#include "Cintex/Cintex.h"

// HDF5 includes
#include "H5Cpp.h"

namespace po = boost::program_options;
using namespace H5;
using namespace std;

/// Time resolution. Obtained by accumulating all times of hits in a run.
/// Using the range across all hits in a run gets zero timing resolution.
/// Offset every times by the earliest hits in an event instead!
//~ const float tmin = 218483.890625;
//~ const float tmax = 228357.421875;
//~ const float tbin = (tmax-tmin)/255;

//------------------------------------------------------------------------------
// To write to HDF5, a contiguous memory block for data is required.
// This class allocates a big linear chunk of memory and records data shape.
// At this moment the classes only support one or four bytes unsigned numbers.
// In the case of label class, integer is also included for PDG values.
template <class T>
class ContiguousArray4D
{
public:
  ContiguousArray4D(int, int, int, int);
  ~ContiguousArray4D();
  
  int fNEntries;
  int fNChannels;
  int fHeight;
  int fWidth;
  DataType fH5Type;
  T**** fData;
  
private:
  void  SetH5Type();
  
  T***  B; ///< B, C, and D are auxiliary arrays.
  T**   C; ///< B, C, and D are auxiliary arrays.
  T*    D; ///< B, C, and D are auxiliary arrays.
};

template <class T>
ContiguousArray4D<T>::ContiguousArray4D(int nentry, int nchannel, int height, int width):
fNEntries(nentry), fNChannels(nchannel), fHeight(height), fWidth(width)
{
  fData  = new T***[fNEntries];
  B = new T**[fNEntries*fNChannels];
  C = new T*[fNEntries*fNChannels*fHeight];
  D = new T[fNEntries*fNChannels*fHeight*fWidth];
  for(int i = 0; i < fNEntries; i++) {
    for(int j = 0; j < fNChannels; j++) {
      for(int k = 0; k < fHeight; k++) {
        C[fHeight*(fNChannels*i+j)+k] = D+(fHeight*(fNChannels*i+j)+k)*fWidth;
      }
      B[fNChannels*i+j] = C+fHeight*(fNChannels*i+j);
    }
    fData[i] = B+fNChannels*i;
  }
  
  SetH5Type();
}

template <class T>
ContiguousArray4D<T>::~ContiguousArray4D()
{
  delete [] D;
  delete [] C;
  delete [] B;
  delete [] fData;
}

template <class T>
void ContiguousArray4D<T>::SetH5Type()
{
  /// Default to unsigned char.
  fH5Type = PredType::STD_U8LE;
  /// Options for unsignd integers with other sizes.
  if(is_same<T, unsigned int>::value) fH5Type = PredType::STD_U32LE;
}
//--end of ContiguousArray4D definition-----------------------------------------
// define simple 2D array for labels--------------------------------------------
template <class T>
class ContiguousArray2D
{
  public:
  ContiguousArray2D(int, int);
  ~ContiguousArray2D();
  
  int fNEntries, fNKeys;
  DataType fH5Type;
  T** fData;
  
private:
  void  SetH5Type();
  
  T*    D; ///< D is an auxiliary array.
};

template <class T>
ContiguousArray2D<T>::ContiguousArray2D(int nentry, int nkeys):
fNEntries(nentry), fNKeys(nkeys)
{
  fData  = new T*[fNEntries];
  D = new T[fNEntries*fNKeys];
  for(int i = 0; i < fNEntries; i++)
    fData[i] = D + fNKeys*i;
      
  SetH5Type();
}

template <class T>
ContiguousArray2D<T>::~ContiguousArray2D()
{
  delete [] D;
  delete [] fData;
}

template <class T>
void ContiguousArray2D<T>::SetH5Type()
{
  /// Default to unsigned char.
  fH5Type = PredType::STD_U8LE;
  /// Options for unsignd integers with other sizes.
  if(is_same<T, unsigned int>::value) fH5Type = PredType::STD_U32LE;
}
//--end of ContiguousArray2D definition-----------------------------------------
// define simple 1D array for labels--------------------------------------------
template <class T>
class ContiguousArray1D
{
public:
  ContiguousArray1D(int);
  ~ContiguousArray1D();
  
  int fNEntries;
  DataType fH5Type;
  T* fData;
  
private:
  void  SetH5Type();
};

template <class T>
ContiguousArray1D<T>::~ContiguousArray1D()
{
  delete [] fData;
}

template <class T>
ContiguousArray1D<T>::ContiguousArray1D(int nentry):
fNEntries(nentry)
{
  fData  = new T[fNEntries];
  SetH5Type();
}

template <class T>
void ContiguousArray1D<T>::SetH5Type()
{
  /// Default to unsigned char.
  fH5Type = PredType::STD_U8LE;
  /// Options for unsignd integers with other sizes.
  if(is_same<T, unsigned int>::value) fH5Type = PredType::STD_U32LE;
  /// Options for signd integers
  if(is_same<T, int>::value) fH5Type = PredType::STD_I32LE;
}
//--end of ContiguousArray1D definition-----------------------------------------

//------------------------------------------------------------------------------
// definition of data containers
//------------------------------------------------------------------------------
// data shape is (# entries) x (# channels) x (# heights) x (# widths)
template <class T>
class PixelMapData : public ContiguousArray4D<T>
{
public:
  PixelMapData(int a, int b, int c, int d) : ContiguousArray4D<T>(a, b, c, d) {};
  void FillCharge(cvn::TrainingData*, int, int, int); ///< First integer entry, second cell, third plane.
  void FillTime(cvn::TrainingData*, int, int, int, float); ///< First integer entry, second cell, third plane. Last time offset.
private:
  unsigned char DigitizeTime(float, float);
};

template <class T>
void PixelMapData<T>::FillCharge(cvn::TrainingData* data, int entry, int cell, int plane)
{
  const int ncells = data->fPMap.fNCell;
  /// Check index range.
  if(entry >= this->fNEntries || cell >= this->fHeight || plane >= this->fWidth) return;
  this->fData[entry][0][cell][plane] = cvn::ConvertToChar(data->fPMap.fPEX.at(cell + ncells*plane), true);
  this->fData[entry][2][cell][plane] = cvn::ConvertToChar(data->fPMap.fPEY.at(cell + ncells*plane), true);
}

template <class T>
void PixelMapData<T>::FillTime(cvn::TrainingData* data, int entry, int cell, int plane, float toffset)
{
  const int ncells = data->fPMap.fNCell;
  /// Check index range.
  if(entry >= this->fNEntries || cell >= this->fHeight || plane >= this->fWidth) return;
  this->fData[entry][1][cell][plane] = DigitizeTime(data->fPMap.fTX.at(cell + ncells*plane), toffset);
  this->fData[entry][3][cell][plane] = DigitizeTime(data->fPMap.fTY.at(cell + ncells*plane), toffset);
}

template <class T>
unsigned char PixelMapData<T>::DigitizeTime(float time, float toffset)
{
  /// In an independent timing study, time span is within about [0..350].
  float digtime = ceil(time-toffset);
  if(digtime > 255) return 255;
  if(digtime < 0) return 0;
  return digtime;
}

//------------------------------------------------------------------------------
// Here the assumption is made that the label shape is (N,),
// where N is the number of entries.
template <class T>
class LabelData : public ContiguousArray1D<T>
{
public:
  LabelData(int a) : ContiguousArray1D<T>(a) {};
  void FillLabel(cvn::TrainingData*, int, T);
};

template <class T>
void LabelData<T>::FillLabel(cvn::TrainingData* data, int entry, T val)
{
  /// Check index range.
  if(entry >= this->fNEntries) return;
  this->fData[entry] = val;
}

//------------------------------------------------------------------------------
// The event id has shape (N,5),
// where N is the number of entries.
template <class T>
class EventIDData : public ContiguousArray2D<T>
{
public:
  EventIDData(int a) : ContiguousArray2D<T>(a, 5) {};
  void FillID(cvn::TrainingData*, int);
};

template <class T>
void EventIDData<T>::FillID(cvn::TrainingData* data, int entry)
{
  /// Check index range.
  if(entry >= this->fNEntries) return;
  this->fData[entry][0] = data->fRun;
  this->fData[entry][1] = data->fSubrun;
  this->fData[entry][2] = data->fCycle;
  this->fData[entry][3] = data->fEvt;
  this->fData[entry][4] = data->fSubevt;
}

//------------------------------------------------------------------------------
// data write out
void HDF5Fill(const H5std_string& FILE_NAME,
              PixelMapData<unsigned char>& data,
              LabelData<unsigned char>&    mode_label,
              LabelData<int>&              pdg_label,
              EventIDData<unsigned int>&   id)
{
  try {
    // Turn off the auto-printing when failure occurs so that we can
    // handle the errors appropriately
    Exception::dontPrint();
    
    // Create a new file using the default property lists.
    H5File file(FILE_NAME, H5F_ACC_TRUNC);
    
    // Create the data space for the dataset.
    hsize_t dims[4];               // dataset dimensions
    dims[0] = data.fNEntries;
    dims[1] = data.fNChannels;
    dims[2] = data.fHeight;
    dims[3] = data.fWidth;
    DataSpace dataspace(4, dims);
    
    // Create the dataset.
    DataSet dataset = file.createDataSet("data", data.fH5Type, dataspace);
    
    // Write the data to the dataset using default memory space, file
    // space, and transfer properties.
    dataset.write(&data.fData[0][0][0][0], data.fH5Type);
    
    // Create the data space for interaction mode labels.
    hsize_t mldims[1];               // label dimensions
    mldims[0] = mode_label.fNEntries;
    DataSpace modelabelspace(1, mldims);
    
    // Create the label dataset.
    DataSet mldataset = file.createDataSet("interaction_mode", mode_label.fH5Type, modelabelspace);
    mldataset.write(&mode_label.fData[0], mode_label.fH5Type);
    
    // Create the data space for pdg labels.
    hsize_t pldims[1];               // label dimensions
    pldims[0] = pdg_label.fNEntries;
    DataSpace pdglabelspace(1, pldims);
    
    // Create the label dataset.
    DataSet pldataset = file.createDataSet("pdg", pdg_label.fH5Type, pdglabelspace);
    pldataset.write(&pdg_label.fData[0], pdg_label.fH5Type);
    
    // Create the data space for the ids.
    hsize_t idims[2];               // label dimensions
    idims[0] = id.fNEntries;
    idims[1] = id.fNKeys;
    DataSpace idspace(2, idims);
    
    // Create the id dataset.
    DataSet idataset = file.createDataSet("id", id.fH5Type, idspace);
    idataset.write(&id.fData[0][0], id.fH5Type);
  }
  catch(FileIException error) { // catch failure caused by the H5File operations
    error.printError();
    return;
  }
}


void fill(po::variables_map& opts)
{
  /// Load predefined labels and functions for retrieving data.
  ROOT::Cintex::Cintex::Enable();
  gSystem->Load("libCVNFunc_dict");
  
  /// Name of the data tree
  TChain chain(opts["TreeName"].as<string>().c_str());
  
  /// Open the input file (list)
  string input = opts["InputFile"].as<string>();
  
  if (boost::ends_with(input,".list")) {
    std::ifstream list_file(input.c_str());
    if (!list_file.is_open()) {
      std::cout << "Could not open " << input << std::endl;
      exit(1);
    }

    std::string ifname;
    while (list_file>>ifname)
      chain.Add(ifname.c_str());
      
  }//end if list file
  
  else if  (boost::ends_with(input,".root")) {
    chain.Add(input.c_str());
  }//end if root file
  else {
    cerr << "File extension is not recognised." << endl;
    exit(1);
  }
  
  /// Event dump container. Note that the event dump contains all possible
  /// information, including that not used by CVN, ex. time of hits.
  cvn::TrainingData* data = new cvn::TrainingData();
  chain.SetBranchAddress(opts["TrainingDataBranchName"].as<string>().c_str(), &data);
  
  unsigned int entries = chain.GetEntries();
  // Resize the number of entries to run through.
  entries = min<unsigned int>(entries, opts["NEvents"].as<unsigned int>());
  if(entries <= 0){
    std::cout << "Error: Input tree has no entries." << std::endl;
    exit(4);
  }

  /// Traning data need to be shuffled. Realizing this by making an ordered list
  /// of consecutive numbers and shuffle them.
  std::srand ( unsigned ( std::time(0) ) );
  std::vector<unsigned int> shuffled;
  for (unsigned int i = 0; i < entries; ++i)
  {
    shuffled.push_back(i);
  }
  std::random_shuffle( shuffled.begin(), shuffled.end() );
  
  /// Figure out the size of the train and test samples
  /// Call a 'block' a particular set of one test and nTrainPerTest train
  int blockSize = opts["NTrainPerTest"].as<int>() + 1;
  /// number of test is the number of blocks, using integer division
  int nTest     = 1 + entries / blockSize;
  /// number of training samples is number of blocks times train per test
  int nTrain    = entries / blockSize * opts["NTrainPerTest"].as<int>();
  /// Add on the entries from the last, potentially partial block, minus test
  if (entries % blockSize) nTrain += entries % blockSize - 1;
  
  /// Create arrays to hold image data. In these case there are 4 channels
  /// (features), namely, energy and time in x and y views.
  /// This data structure assumes that the data shapes are the same for all
  /// data.
  chain.GetEntry(0); // for getting the image dimensions
  const int nChannels = 4;
  const int planes = data->fPMap.fNPlane/2;
  const int cells = data->fPMap.fNCell;
  /// Data containers
  PixelMapData<unsigned char> trainingSample(nTrain, nChannels, cells, planes);
  PixelMapData<unsigned char> testSample(nTest, nChannels, cells, planes);
  EventIDData<unsigned int>   trainingID(nTrain);
  EventIDData<unsigned int>   testID(nTest);
  LabelData<unsigned char>    trainingModeLabel(nTrain);
  LabelData<unsigned char>    testModeLabel(nTest);
  LabelData<int>    trainingPdgLabel(nTrain);
  LabelData<int>    testPdgLabel(nTest);
  
  
  /// Start event loop and extract data
  int iTrain = 0;
  int iTest  = 0;
  for(unsigned int iEntry = 0; iEntry < entries; ++iEntry)
  {
    chain.GetEntry(shuffled[iEntry]);
    
    /// In order to offset times of hits by the time of the earliest hit,
    /// 2 passes of loops are needed.
    if(iEntry % (blockSize))
    {
      vector<float> tx, ty; ///< timing containers
      
      for (int iPlane = 0; iPlane < planes; ++iPlane)
        for (int iCell = 0; iCell < cells; ++iCell) {
          trainingSample.FillCharge(data, iTrain, iCell, iPlane);
          if (data->fPMap.fTX.at(iCell + cells*iPlane) > 1e-3)
            tx.push_back(data->fPMap.fTX.at(iCell + cells*iPlane));
          if (data->fPMap.fTY.at(iCell + cells*iPlane) > 1e-3)
            ty.push_back(data->fPMap.fTY.at(iCell + cells*iPlane));
        }
      
      /// Find timing offset.
      float txmin = numeric_limits<float>::max();
      if(tx.size()) txmin = *min_element(tx.begin(), tx.end());
      float tymin = numeric_limits<float>::max();
      if(ty.size()) tymin = *min_element(ty.begin(), ty.end());
      float tmin  = min(txmin, tymin);
      if(!tx.size() || !ty.size())
      {
        cout << "Warning: At least one view has no timing data!" << endl;
        cout << "The minimum timestamp of views combined is " << tmin << endl;
      }
      /// Second pass...
      for (int iPlane = 0; iPlane < planes; ++iPlane)
        for (int iCell = 0; iCell < cells; ++iCell)
          trainingSample.FillTime(data, iTrain, iCell, iPlane, tmin);
      
      /// Fill event label
      trainingModeLabel.FillLabel(data, iTrain, data->fInt);
      trainingPdgLabel.FillLabel(data, iTrain, data->fNuPdg);
      /// Fill event id
      trainingID.FillID(data, iTrain);
      
      iTrain++;
    }
    else { // Fill test sample.
      vector<float> tx, ty; ///< timing containers
      
      for (int iPlane = 0; iPlane < planes; ++iPlane)
        for (int iCell = 0; iCell < cells; ++iCell) {
          testSample.FillCharge(data, iTest, iCell, iPlane);
          if (data->fPMap.fTX.at(iCell + cells*iPlane) > 1e-3)
            tx.push_back(data->fPMap.fTX.at(iCell + cells*iPlane));
          if (data->fPMap.fTY.at(iCell + cells*iPlane) > 1e-3)
            ty.push_back(data->fPMap.fTY.at(iCell + cells*iPlane));
        }
      
      /// Find timing offset.
      float txmin = numeric_limits<float>::max();
      if(tx.size()) txmin = *min_element(tx.begin(), tx.end());
      float tymin = numeric_limits<float>::max();
      if(ty.size()) tymin = *min_element(ty.begin(), ty.end());
      float tmin  = min(txmin, tymin);
      if(!tx.size() || !ty.size())
      {
        cout << "Warning: At least one view has no timing data!" << endl;
        cout << "The minimum timestamp of views combined is " << tmin << endl;
      }
      /// Second pass...
      for (int iPlane = 0; iPlane < planes; ++iPlane)
        for (int iCell = 0; iCell < cells; ++iCell)
          testSample.FillTime(data, iTest, iCell, iPlane, tmin);
      
      /// Fill test label
      testModeLabel.FillLabel(data, iTest, data->fInt);
      testPdgLabel.FillLabel(data, iTest, data->fNuPdg);
      /// Fill event id
      testID.FillID(data, iTest);
      
      iTest++;
    }
    
    // Print progress.
    int eprog = iEntry + 1;
    if(!(eprog % 100)) cout << eprog << " events finished." << endl;
  } // Done with all data filling.
  
  // In the end, write to HDF5 files.
  HDF5Fill("training_data.h5", trainingSample, trainingModeLabel, trainingPdgLabel, trainingID);
  HDF5Fill("test_data.h5", testSample, testModeLabel, testPdgLabel, testID);
}


int main(int argc, char* argv[])
{
  try {
    /// Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
      ("help", "produce help message")
      ("NEvents,n", po::value<unsigned int>()->default_value(UINT_MAX), "Number of events to run through")
      ("InputFile,i", po::value<string>()->default_value("/nova/ana/users/slin/temp/cvn_test/event_dump_round2/test_input.list"), "file name of input file (.root) or list (.list)")
      ("TreeName", po::value<string>()->default_value("cvndump/CVNTrainTree"), "name of the data tree")
      ("TrainingDataBranchName", po::value<string>()->default_value("train"), "name of the training data branch")
      ("NTrainPerTest", po::value<int>()->default_value(4), "Ratio of training to test data. Ex. 4 means 80/20 split.")
    ;
    
    po::variables_map opts;
    po::store(po::parse_command_line(argc, argv, desc), opts);
    po::notify(opts);
    
    if (opts.count("help")) {
      cout << desc << endl;
      return 0;
    }
    
    /// Check file existence.
    if(!boost::filesystem::exists(opts["InputFile"].as<string>())) {
      cerr << "File " << opts["InputFile"].as<string>() << " does not exist." << endl;
      return 1;
    }
    
    fill(opts);
  }
  catch(exception& e) {
    cerr << "error: " << e.what() << endl;
    return 1;
  }
  catch(...) {
    cerr << "Exception of unknown type!" << endl;
  }
  
  return 0;
}
