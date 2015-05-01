#include "UnitTest++.h"
#include "MultiResolutionImage.h"
#include "MultiResolutionImageReader.h"
#include "MultiResolutionImageWriter.h"
#include <iostream>
#include "core/filetools.h"
#include "core/PathologyEnums.h"
#include "TestData.h"
#include "config/pathology_config.h"

using namespace UnitTest;
using namespace std;
using namespace pathology;

namespace
{
  SUITE(MultiResolutionImageInterface)
  {
	  
    TEST(TestCanOpen)
    {
      MultiResolutionImageReader test;
      MultiResolutionImage* img = test.open(g_dataPath + "/images/OpenSlideInterfaceTestImage.svs");
  		CHECK(img);
      CHECK(img->valid());
      delete img;
	  }

    TEST(TestGetNumberOfLevels)
    {
      MultiResolutionImageReader test;
      MultiResolutionImage* img = test.open(g_dataPath + "/images/OpenSlideInterfaceTestImage.svs");
      unsigned int nrLevels = img->getNumberOfLevels();
  		CHECK_EQUAL(6, nrLevels);
      delete img;
	  }

    TEST(TestGetDimensionsLevel0)
    {
      MultiResolutionImageReader test;
      vector<unsigned long long> dims;
      MultiResolutionImage* img = test.open(g_dataPath + "/images/OpenSlideInterfaceTestImage.svs");
      dims = img->getDimensions();
  		CHECK_EQUAL(26112, dims[0]);
      CHECK_EQUAL(13824, dims[1]);
      delete img;
	  }

    TEST(TestGetDimensionsLevel2)
    {
      MultiResolutionImageReader test;
      vector<unsigned long long> dims;
      MultiResolutionImage* img = test.open(g_dataPath + "/images/OpenSlideInterfaceTestImage.svs");
      dims = img->getLevelDimensions(2);
  		CHECK_EQUAL(6528, dims[0]);
      CHECK_EQUAL(3456, dims[1]);
      delete img;
	  }

    TEST(TestgetRawRegionUInt32)
    {
      MultiResolutionImageReader test;
      MultiResolutionImage* img = test.open(g_dataPath + "/images/OpenSlideInterfaceTestImage.tif");
      unsigned int* data = new unsigned int[512*512*3];
      img->getRawRegion<unsigned int>(13824,11776,512,512,0,data);
      CHECK_EQUAL(231, (int)data[(512*255+255)*3]);
      CHECK_EQUAL(181, (int)data[(512*255+255)*3+1]);
      CHECK_EQUAL(207, (int)data[(512*255+255)*3+2]);
      delete[] data;
      data = NULL;
      delete img;
	  }

    TEST(TestgetRawRegionUChar)
    {
      MultiResolutionImageReader test;
      MultiResolutionImage* img = test.open(g_dataPath + "/images/OpenSlideInterfaceTestImage.tif");
      unsigned char* data = new unsigned char[512*512*3];
      img->getRawRegion<unsigned char>(13824,11776,512,512,0,data);
      CHECK_EQUAL(231, (int)data[(512*255+255)*3]);
      CHECK_EQUAL(181, (int)data[(512*255+255)*3+1]);
      CHECK_EQUAL(207, (int)data[(512*255+255)*3+2]);
      delete[] data;
      data = NULL;
      delete img;
	  }
    
    TEST(TestgetRawRegionUCharOpenSlide)
    {
      MultiResolutionImageReader test;
      MultiResolutionImage* img = test.open(g_dataPath + "/images/OpenSlideInterfaceTestImage.svs");
      unsigned char* data = new unsigned char[512*512*4];
      img->getRawRegion<unsigned char>(13824,11776,512,512,0,data);
      CHECK_EQUAL(207, (int)data[(512*255+255)*4]);
      CHECK_EQUAL(181, (int)data[(512*255+255)*4+1]);
      CHECK_EQUAL(231, (int)data[(512*255+255)*4+2]);
      CHECK_EQUAL(255, (int)data[(512*255+255)*4+3]);
      delete[] data;
      data = NULL;
      delete img;
    }

    TEST(TestgetRawRegionFloatOpenSlide)
    {
      MultiResolutionImageReader test;
      MultiResolutionImage* img = test.open(g_dataPath + "/images/OpenSlideInterfaceTestImage.svs");
      float* data = new float[512*512*4];
      img->getRawRegion<float>(13824,11776,512,512,0,data);
      CHECK_CLOSE(207., data[(512*255+255)*4],0.01);
      CHECK_CLOSE(181., data[(512*255+255)*4+1],0.01);
      CHECK_CLOSE(231., data[(512*255+255)*4+2],0.01);
      CHECK_CLOSE(255., data[(512*255+255)*4+3],0.01);
      delete[] data;
      data = NULL;
      delete img;
    }
    
    TEST(TestReadWriteSingleChannelFloat)
    {
      MultiResolutionImageReader testRead;
      MultiResolutionImageWriter testWrite;
      MultiResolutionImage* img = testRead.open(g_dataPath + "/images/OpenSlideInterfaceTestImage.tif");
      testWrite.openFile(g_dataPath + "/images/OpenSlideInterfaceTestImageFloatOut.tif");
      testWrite.setTileSize(256);
      testWrite.setCompression(LZW);
      testWrite.setDataType(Float);
      testWrite.setColorType(Monochrome);
      testWrite.writeImageInformation(512,512);
      for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
          float* data = new float[256*256*3];
          float* dataR = new float[256*256];
          img->getRawRegion<float>(13824+j*256,11776+i*256,256,256,0,data);
          for (int k = 0; k < 256*256; ++k) {
            dataR[k] = data[k*3]+0.5;
          }
          testWrite.writeBaseImagePart((void*)dataR);
          delete[] data;
          delete[] dataR;
          data = NULL;
          dataR = NULL;
        }
      }
      testWrite.finishImage();
      unsigned char* dataOrg = new unsigned char[256*256*3];
      img->getRawRegion(13824,11776,256,256,0,dataOrg);
      MultiResolutionImageReader testRead2;
      img = testRead2.open(g_dataPath + "/images/OpenSlideInterfaceTestImageFloatOut.tif");
      float* dataWritten = new float[256*256];
      img->getRawRegion(0,0,256,256,0,dataWritten);
      float test1 = (float)(dataOrg[(256*127+127)*3]+0.5);
      float test2 = dataWritten[256*127+127];
      CHECK_EQUAL(test1, test2);
      delete[] dataOrg, dataWritten;
    }

    TEST(TestReadWriteSingleChannelUintLevel2)
    {
      MultiResolutionImageReader testRead;
      MultiResolutionImageWriter testWrite;
      MultiResolutionImage* img = testRead.open(g_dataPath + "/images/OpenSlideInterfaceTestImage.tif");
      testWrite.openFile(g_dataPath + "/images/OpenSlideInterfaceTestImageUint32Out.tif");
      testWrite.setTileSize(256);
      testWrite.setCompression(LZW);
      testWrite.setDataType(UInt32);
      testWrite.setColorType(Monochrome);
      testWrite.writeImageInformation(512,512);
      for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
          unsigned int* data = new unsigned int[256*256*3];
          unsigned int* dataR = new unsigned int[256*256];
          img->getRawRegion<unsigned int>(4500+i*256,6500+j*256,256,256,2,data);
          for (int k = 0; k < 256*256; ++k) {
            dataR[k] = data[k*3]*20000;
          }
          testWrite.writeBaseImagePart((void*)dataR);
          delete data;
          data = NULL;
        }
      }
      testWrite.finishImage();
      unsigned char* dataOrg = new unsigned char[256*256*3];
      img->getRawRegion(4500,6500,256,256,2,dataOrg);
      delete img;
      MultiResolutionImageReader testRead2;
      img = testRead2.open(g_dataPath + "/images/OpenSlideInterfaceTestImageUint32Out.tif");
      unsigned int* dataWritten = new unsigned int[256*256];
      img->getRawRegion(0,0,256,256,0,dataWritten);
      unsigned int test1 = (unsigned int)dataOrg[(256*127+127)*3]*20000;
      unsigned int test2 = (unsigned int)dataWritten[256*127+127];
      CHECK_EQUAL(test1, test2);
      delete[] dataOrg, dataWritten;
      delete img;
    }

    TEST(TestReadWriteSingleChannelUint16Level2)
    {
      MultiResolutionImageReader testRead;
      MultiResolutionImageWriter testWrite;
      MultiResolutionImage* img = testRead.open(g_dataPath + "/images/OpenSlideInterfaceTestImage.tif");
      testWrite.openFile(g_dataPath + "/images/OpenSlideInterfaceTestImageUint32Out.tif");
      testWrite.setTileSize(256);
      testWrite.setCompression(LZW);
      testWrite.setDataType(UInt16);
      testWrite.setColorType(Monochrome);
      testWrite.writeImageInformation(512,512);
      for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
          unsigned short* data = new unsigned short[256*256*3];
          unsigned short* dataR = new unsigned short[256*256];
          img->getRawRegion<unsigned short>(4500+i*256,6500+j*256,256,256,2,data);
          for (int k = 0; k < 256*256; ++k) {
            dataR[k] = data[k*3]*20000;
          }
          testWrite.writeBaseImagePart((void*)dataR);
          delete data;
          data = NULL;
        }
      }
      testWrite.finishImage();
      unsigned short* dataOrg = new unsigned short[256*256*3];
      img->getRawRegion(4500,6500,256,256,2,dataOrg);
      delete img;
      MultiResolutionImageReader testRead2;
      img = testRead2.open(g_dataPath + "/images/OpenSlideInterfaceTestImageUint32Out.tif");
      unsigned short* dataWritten = new unsigned short[256*256];
      img->getRawRegion(0,0,256,256,0,dataWritten);
      unsigned short test1 = (unsigned short)dataOrg[(256*127+127)*3]*20000;
      unsigned short test2 = (unsigned short)dataWritten[256*127+127];
      CHECK_EQUAL(test1, test2);
      delete[] dataOrg, dataWritten;
      delete img;
    }

    TEST(TestReadWriteMultiRes)
    {
      MultiResolutionImageReader testRead;
      MultiResolutionImageWriter testWrite;
      MultiResolutionImage* img = testRead.open(g_dataPath + "/images/OpenSlideInterfaceTestImage.tif");
      testWrite.openFile(g_dataPath + "/images/OpenSlideInterfaceMultiResOut.tif");
      testWrite.setTileSize(512);
      testWrite.setCompression(JPEG);
      testWrite.setJPEGQuality(30);
      testWrite.setDataType(UChar);
      testWrite.setColorType(RGB);
      vector<unsigned long long> dims = img->getLevelDimensions(1);
      testWrite.writeImageInformation(dims[0],dims[1]);
      for (int y =0; y < dims[1]; y+=512) {
        for (int x =0; x < dims[0]; x+=512) {
          unsigned char* data = new unsigned char[512*512*3];
          img->getRawRegion(x*img->getLevelDownsample(1),y*img->getLevelDownsample(1),512,512,1,data);
          testWrite.writeBaseImagePart((void*)data);
          delete data;
          data = NULL;
        }
      }
      testWrite.finishImage();
      unsigned char* dataOrg = new unsigned char[512*512*3];
      img->getRawRegion(1000,1000,512,512,1,dataOrg);
      delete img;
      MultiResolutionImageReader testRead2;
      img = testRead2.open(g_dataPath + "/images/OpenSlideInterfaceMultiResOut.tif");
      unsigned char* dataWritten = new unsigned char[512*512*3];
      img->getRawRegion(1000,1000,512,512,0,dataWritten);
      unsigned char test1 = dataOrg[(256*127+127)*3+1];
      unsigned char test2 = dataWritten[(256*127+127)*3+1];
      CHECK_EQUAL(test1, test2);
      delete[] dataOrg, dataWritten;
      delete img;
    }

    TEST(TestReadWriteMultiResOneGo)
    {
      MultiResolutionImageReader testRead;
      MultiResolutionImageWriter testWrite;
      testWrite.setTileSize(512);
      testWrite.setCompression(LZW);
      MultiResolutionImage* img = testRead.open(g_dataPath + "/images/OpenSlideInterfaceTestImage.tif");
      testWrite.writeImageToFile(img, g_dataPath + "/images/OpenSlideInterfaceMultiResOutOneGo.tif");
      delete img;
    }

    TEST(TestReadWriteMultiResMono)
    {
      MultiResolutionImageReader testRead;
      MultiResolutionImageWriter testWrite;
      MultiResolutionImage* img = testRead.open(g_dataPath + "/images/OpenSlideInterfaceTestImage.svs");
      testWrite.openFile(g_dataPath + "/images/OpenSlideInterfaceMultiResOutProc.tif");
      testWrite.setTileSize(512);
      testWrite.setCompression(LZW);
      testWrite.setDataType(UChar);
      testWrite.setColorType(Monochrome);
      vector<unsigned long long> dims = img->getLevelDimensions(1);
      testWrite.writeImageInformation(dims[0],dims[1]);
      for (int y =0; y < dims[1]; y+=512) {
        for (int x =0; x < dims[0]; x+=512) {
          unsigned char* data = new unsigned char[512*512*4];
          unsigned char* procData = new unsigned char[512*512];
          img->getRawRegion(x*img->getLevelDownsample(1),y*img->getLevelDownsample(1),512,512,1,data);
          for (int ty=0; ty < 512; ++ty) {
            for (int tx=0; tx < 512; ++tx) {
              *(procData+(ty*512)+tx) = (unsigned char)((*(data+(ty*512*4)+tx*4) + *(data+(ty*512*4)+tx*4+3)))/2.;
            }
          }
          testWrite.writeBaseImagePart((void*)procData);
          delete procData;
          procData = NULL;
          delete data;
          data = NULL;
        }
      }
      testWrite.finishImage();
      delete img;
    }

    TEST(TestReadWriteMultiResJPEG2000)
    {
      MultiResolutionImageReader testRead;
      MultiResolutionImageWriter testWrite;
      MultiResolutionImage* img = testRead.open(g_dataPath + "/images/OpenSlideInterfaceTestImage.tif");
      testWrite.openFile(g_dataPath + "/images/OpenSlideInterfaceMultiResOutJPEG2000.tif");
      testWrite.setTileSize(512);
      testWrite.setCompression(JPEG2000_LOSSY);
      testWrite.setJPEGQuality(10);
      testWrite.setDataType(UChar);
      testWrite.setColorType(RGB);
      vector<unsigned long long> dims = img->getDimensions();
      testWrite.writeImageInformation(dims[0],dims[1]);
      for (int y =0; y < dims[1]; y+=512) {
        for (int x =0; x < dims[0]; x+=512) {
          unsigned char* data = new unsigned char[512*512*3];
          img->getRawRegion(x,y,512,512,0,data);
          testWrite.writeBaseImagePart((void*)data);
          delete data;
          data = NULL;
        }
      }
      testWrite.finishImage();
      delete img;
    }
  }
  
  SUITE(VSISupport)
  {

	  TEST(ReadingVSI)
	  {
		  MultiResolutionImageReader test;
		  MultiResolutionImage* img = test.open(g_dataPath + "\\images\\T13-02239-II.vsi");
  		CHECK(img);
      CHECK(img->valid());
      delete img;
	  }
	  TEST(getRawRegionFromVSI)
	  {
		  MultiResolutionImageReader test;
		  MultiResolutionImage* img = test.open(g_dataPath + "\\images\\T13-02239-II.vsi");
      unsigned char* testData = new unsigned char[512*512*3];
      img->getRawRegion(3608,9752,512,512,0,testData);
      MultiResolutionImageWriter testWrite;
      testWrite.openFile(g_dataPath + "/images/VSIImageReadTest.tif");
      testWrite.setTileSize(512);
      testWrite.setCompression(LZW);
      testWrite.setDataType(UChar);
      testWrite.setColorType(RGB);
      testWrite.writeImageInformation(512,512);
      testWrite.writeBaseImagePart((void*)testData);
      testWrite.finishImage();
      delete[] testData;
      delete img;
	  }
    
    TEST(ReadLosslessJPEG) 
    {
      MultiResolutionImageReader test;
      if (core::fileExists("D:\\Experiments\\Papers\\PatholMRI\\Pathologie\\Biopten\\Original\\VSI\\T13-05194-I.vsi")) {
        MultiResolutionImage* img = test.open("D:\\Experiments\\Papers\\PatholMRI\\Pathologie\\Biopten\\Original\\VSI\\T13-05194-I.vsi");
        MultiResolutionImageWriter testWrite;
        testWrite.openFile(g_dataPath + "/images/VSIImageReadLosslessTest.tif");
        testWrite.setTileSize(512);
        testWrite.setCompression(LZW);
        testWrite.setDataType(UChar);
        testWrite.setColorType(RGB);
        vector<unsigned long long> dims = img->getDimensions();
        testWrite.writeImageInformation(dims[0],dims[1]);        
        for (int y=0; y < dims[1]; y+=512) {
          cout << y << "/" << dims[1] << endl;
          for (int x=0; x < dims[0]; x+=512) {
            unsigned char* data = new unsigned char[512*512*3];
            img->getRawRegion(x,y,512,512,0,data);
            testWrite.writeBaseImagePart((void*)data);
            delete data;
            data = NULL;
          }
        }
        testWrite.finishImage();
        delete img;
      }
    }
  }

  SUITE(LIFSupport)
  { 
    TEST(TestCanOpenLIF)
    {
      MultiResolutionImageReader test;
      MultiResolutionImage* img = test.open(g_dataPath + "/images/TestLeicaLIF.lif");
      delete img;
	  }

    TEST(TestReadFromLIF)
    {
      MultiResolutionImageReader test;
      MultiResolutionImage* img = test.open(g_dataPath + "/images/TestLeicaLIF.lif");
      MultiResolutionImageWriter testWrite;
      testWrite.openFile(g_dataPath + "/images/LIFTestImageUint16Out.tif");
      testWrite.setTileSize(256);
      testWrite.setCompression(LZW);
      testWrite.setDataType(UInt16);
      testWrite.setColorType(Monochrome);
      testWrite.writeImageInformation(512,512);
      for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
          unsigned short* data = new unsigned short[256*256*2];
          unsigned short* dataR = new unsigned short[256*256];
          img->getRawRegion<unsigned short>(4000+x*256,4000+y*256,256,256,0,data);
          for (int k = 0; k < 256*256; ++k) {
            dataR[k] = data[2*k];
          }
          testWrite.writeBaseImagePart((void*)dataR);
          delete data;
          delete dataR;
          data = NULL;
          dataR = NULL;
        }
      }
      testWrite.finishImage();
      delete img;
    }

    TEST(TestReadFromLIFToMultiResIndexed)
    {

      MultiResolutionImageReader test;
      MultiResolutionImage* img = test.open(g_dataPath + "/images/TestLeicaLIF.lif");
      MultiResolutionImageWriter testWrite;
      testWrite.openFile(g_dataPath + "/images/LIFTestImageMultiResOutIndexed.tif");
      testWrite.setTileSize(256);
      testWrite.setCompression(LZW);
      testWrite.setDataType(UInt16);
      testWrite.setColorType(Indexed);
      testWrite.setNumberOfIndexedColors(2);
      std::vector<unsigned long long> L0Dims = img->getDimensions();
      testWrite.writeImageInformation((int)((L0Dims[0]/256)+0.5)*256, 256*(int)((L0Dims[1]/256)+0.5));
      for (int y = 0; y < (int)((L0Dims[1]/256)+0.5); ++y) {
        for (int x = 0; x < (int)((L0Dims[0]/256)+0.5); ++x) {
          unsigned short* data = new unsigned short[256*256*2];
          img->getRawRegion<unsigned short>(x*256,y*256,256,256,0,data);
          testWrite.writeBaseImagePart((void*)data);
          delete data;
          data = NULL;
        }
      }
      testWrite.finishImage();
      delete img;
    }

    TEST(TestReadFromLIFToMultiResIndexedOneGo)
    {

      MultiResolutionImageReader test;
      MultiResolutionImage* img = test.open(g_dataPath + "/images/TestLeicaLIF.lif");
      MultiResolutionImageWriter testWrite;
      testWrite.setTileSize(256);
      testWrite.setCompression(LZW);
      testWrite.writeImageToFile(img, g_dataPath + "/images/LIFTestImageMultiResOutIndexedOneGo.tif");
      delete img;

      img = test.open(g_dataPath + "/images/LIFTestImageMultiResOutIndexedOneGo.tif");
      double min = img->getMinValue();
      double max = img->getMaxValue();

    }
  }

  /*
  SUITE(MRXSSupport)
  {

	  TEST(ReadingMRXS)
	  {
		  MultiResolutionImageReader test;
		  test.open("E:\\Temp\\T10-00014-I1-1_2.0.mrxs");
      test.close();
	  }

	  TEST(getRawRegionFromMRXS)
	  {
		  MultiResolutionImageReader test;
		  test.open("E:\\Temp\\T10-00014-I1-1_2.0.mrxs");
      unsigned char* testData = new unsigned char[512*512*4];
      unsigned char* outData = new unsigned char[512*512*3];
      test.getRawRegion(40000,70000,512,512,0,testData);
      for (int i = 0, j = 0; i < 512*512*4; i+=4, j+=3) {
        outData[j] = testData[i];
        outData[j+1] = testData[i+1];
        outData[j+2] = testData[i+2];
      }
      MultiResolutionImageWriter testWrite;
      testWrite.openFile(g_dataPath + "/images/MRXSTest.tif");
      testWrite.setTileSize(512);
      testWrite.setAperioCompatibility(true);
      testWrite.setCompression(MultiResolutionImageWriter::LZW);
      testWrite.setDataType(MultiResolutionImageWriter::UChar);
      testWrite.setColorType(MultiResolutionImageWriter::RGB);
      testWrite.writeImageInformation(512,512);
      testWrite.writeBaseImagePart((void*)outData);
      testWrite.finishImage();
      delete[] testData;
      delete[] outData;
	  }
  }*/
}