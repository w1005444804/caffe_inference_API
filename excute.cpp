#include "mini_caffe.h"

InferNet::InferNet(const IFparams& ifparams){
	Init(ifparams);
}

void InferNet::Init(const IFparams& ifparams){
	//create net
	EnginNetPtr_.reset(new caffe::Net(ifparams.Proto_Path));
	EnginNetPtr_->CopyTrainedLayersFrom(ifparams.Model_Path);
	CPU_ONLY_ = ifparams.CPU_ONLY;
	if (CPU_ONLY_){
		caffe::SetMode(caffe::CPU, 0);
	}
	else {
		caffe::SetMode(caffe::GPU, 0);
	}

	for (int i = 0; i < ifparams.object_names.size(); ++i){
		std::string name = ifparams.object_names[i];
		std::shared_ptr<caffe::Blob> blob = EnginNetPtr_->blob_by_name(name);
		EnginNetPtr_->MarkOutputs({name});
		if (OutPut_.count(name) == 0){
			std::shared_ptr<Datum> datum;
			BlobToDatum(blob, datum);
			OutPut_.insert(std::make_pair(name, datum));
		}
		else {
			printf("the object name: %s has defined: \n", name);
			CHECK(0);
		}
	}
}

void InferNet::Infer(const std::shared_ptr<Datum> input_datum){
	//input blob name is data;
	DatumToBlob(input_datum, EnginNetPtr_->blob_by_name("data"));

	// it is a bug when this dll is used to compile other dll.
	if (CPU_ONLY_){
		caffe::SetMode(caffe::CPU, 0);
	} else {
		caffe::SetMode(caffe::GPU, 0);
	}

	EnginNetPtr_->Forward();

	//configure output
	for (auto it = OutPut_.begin(); it != OutPut_.end(); ++it){
		std::string name = it->first;
		std::shared_ptr<Datum> datum = it->second;
		std::shared_ptr<caffe::Blob> blob = EnginNetPtr_->blob_by_name(name);
		BlobToDatum(blob, datum);
	}
}
/* ------------------------------------------------- */
/* ------------------------------------------------- */
void Inference::Init(IFparams& pParams){
	if (InferNet_ != nullptr){
		delete static_cast<InferNet*>(InferNet_);
	}
	InferNet* IFN = new InferNet();
	IFN->Init(pParams);
	InferNet_ = static_cast<void*>(IFN);
}
Inference::IFReult Inference::Infer(const std::shared_ptr<Datum> input_datum){
	static_cast<InferNet*>(InferNet_)->Infer(input_datum);
	return static_cast<InferNet*>(InferNet_)->GetResult();
}
Inference::~Inference(){
	delete static_cast<InferNet*>(InferNet_);
}