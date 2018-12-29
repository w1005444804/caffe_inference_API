#ifndef  _TNT_H_
#define  _TNT_H_

#include <memory>
#include <map>
#include <vector>

#ifdef _WIN32
#ifdef INFER_EXPORTS
#define INFER_API __declspec(dllexport)
#else
#define INFER_API __declspec(dllimport)
#endif
//added else for support Unix
#else
#define RPN_API
#endif
//#define CPU_ONLY


struct Shape {
	Shape(){}
	Shape(unsigned int n, unsigned int c, unsigned int h, unsigned int w){
		N = n; C = c; H = h; W = w;
	}
	unsigned int count() {
		return N*C*H*W;
	}
	unsigned int N{ 1 };
	unsigned int C{ 1 };
	unsigned int H{ 1 };
	unsigned int W{ 1 };
};

struct Datum {
	Shape shape;
	float* data = nullptr;//NCHW
	Shape out_shape;
	float* outter_data = nullptr;

	Datum() {
		data = nullptr;
	}
	float* Getdata() {
		if (data != nullptr) {
			return data;
		}
		else {
			unsigned int size = shape.count();
			data = new float[size];
			return data;
		}
	}
	void Reshape(Shape s) {
		if (data != nullptr && s.count() > shape.count()) {
			delete[] data;
			data = nullptr;
		}
		shape = s;
	}
	~Datum() {
		if (data != nullptr){
			delete[] data;
			data = nullptr;
		}
	}
};

struct IFparams {
	bool CPU_ONLY{ 1 };
	char* Root_Path;
	char* Model_Path;
	char* Proto_Path;
	Shape shape;
	std::vector<std::string> object_names;
};

class INFER_API Inference {
public:
	typedef std::map<std::string, std::shared_ptr<Datum>> IFReult;

	Inference(){};

	virtual~Inference();

	void Init(IFparams& ifparams);

	IFReult Infer(const std::shared_ptr<Datum> input_datum);

private:

	void* InferNet_ = nullptr;

};
#endif //_MTCNN_H_