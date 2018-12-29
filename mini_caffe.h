#ifndef  _MINI_CAFFE_H_
#define  _MINI_CAFFE_H_
#pragma once

#include "Infer.h"

#include "caffe/caffe.hpp"
#include "caffe/c_api.h"

#include <fstream>

using SizeVector = std::vector<size_t>;

#define CHECK_SUCCESS(condition)                  \
  if ((condition) != 0) {                         \
    printf("CHECK (" #condition ") failed\n");    \
    printf("%s\n", CaffeGetLastError());          \
    exit(-1);                                     \
      }

#define CHECK(condition)                          \
  if (!(condition)) {                             \
    printf("CHECK (" #condition ") failed\n");    \
    exit(-1);                                     \
      }

bool compare_shape(const Shape& S1, const Shape& S2) {
	if (S1.W != S2.W || S1.H != S2.H ||
		S1.C != S2.C || S1.N != S2.N) {
		return 1;
	}
	return 0;
}

void DatumToBlob(const std::shared_ptr<Datum> datum, std::shared_ptr<caffe::Blob> blobptr){
	std::vector<int> shape;
	shape.reserve(4);
	shape.push_back(datum->shape.N);
	shape.push_back(datum->shape.C);
	shape.push_back(datum->shape.H);
	shape.push_back(datum->shape.W);
	blobptr->Reshape(shape);
	float* blob_data = blobptr->mutable_cpu_data();
	std::memcpy(blob_data, datum->Getdata(), datum->shape.count() * sizeof(float));
}

void BlobToDatum(const std::shared_ptr<caffe::Blob> blobptr, std::shared_ptr<Datum> datum){
	const std::vector<int> shape = blobptr->shape();
	Shape newdatumshape;
	const int veclength = shape.size();
	if (shape.size() > 0){
		newdatumshape.N = shape[0];
	}
	if (shape.size() > 1){
		newdatumshape.C = shape[1];
	}
	if (shape.size() > 2){
		newdatumshape.H = shape[2];
	}
	if (shape.size() > 3){
		newdatumshape.W = shape[3];
	}
	datum->out_shape = newdatumshape;
	datum->outter_data = blobptr->mutable_cpu_data();
}

class InferNet {
public:
	explicit InferNet(const IFparams& ifparam);

	InferNet(){};

	virtual~InferNet(){};

	void Init(const IFparams& ifparams);

	void Infer(const std::shared_ptr<Datum> input_datum);

	Inference::IFReult GetResult(){ return OutPut_; };

private:

	std::shared_ptr<caffe::Net> EnginNetPtr_;

	Inference::IFReult OutPut_;

	bool CPU_ONLY_;
};


#endif //_MTCNN_H_