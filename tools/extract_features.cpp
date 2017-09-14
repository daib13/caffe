#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using std::string;
namespace db = caffe::db;

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv);

int main(int argc, char** argv) {
	return feature_extraction_pipeline<float>(argc, argv);
	//  return feature_extraction_pipeline<double>(argc, argv);
}

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv) {
	::google::InitGoogleLogging(argv[0]);
	const int num_required_args = 6;
	if (argc < num_required_args) {
		LOG(ERROR) <<
			"This program takes in a trained network and an input data layer, and then"
			" extract features of the input data produced by the net.\n"
			"Usage: extract_features  pretrained_net_param"
			"  feature_extraction_proto_file  extract_feature_blob_name1[,name2,...]"
			"  save_feature_dataset_name1[,name2,...]  num_mini_batches"
			"  [CPU/GPU] [DEVICE_ID=0]\n"
			"Note: you can extract multiple features in one pass by specifying"
			" multiple feature blob names and dataset names separated by ','."
			" The names cannot contain white space characters and the number of blobs"
			" and datasets must be equal.";
		return 1;
	}
	int arg_pos = num_required_args;

	arg_pos = num_required_args;
	if (argc > arg_pos && strcmp(argv[arg_pos], "GPU") == 0) {
		LOG(ERROR) << "Using GPU";
		int device_id = 0;
		if (argc > arg_pos + 1) {
			device_id = atoi(argv[arg_pos + 1]);
			CHECK_GE(device_id, 0);
		}
		LOG(ERROR) << "Using Device_id=" << device_id;
		Caffe::SetDevice(device_id);
		Caffe::set_mode(Caffe::GPU);
	}
	else {
		LOG(ERROR) << "Using CPU";
		Caffe::set_mode(Caffe::CPU);
	}

	arg_pos = 0;  // the name of the executable
	std::string pretrained_binary_proto(argv[++arg_pos]);

	std::string feature_extraction_proto(argv[++arg_pos]);
	boost::shared_ptr<Net<Dtype> > feature_extraction_net(
		new Net<Dtype>(feature_extraction_proto, caffe::TEST));
	feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);

	std::string extract_feature_blob_names(argv[++arg_pos]);
	std::vector<std::string> blob_names;
	boost::split(blob_names, extract_feature_blob_names, boost::is_any_of(","));

	std::string save_feature_dataset_names(argv[++arg_pos]);
	std::vector<std::string> dataset_names;
	boost::split(dataset_names, save_feature_dataset_names,
		boost::is_any_of(","));
	CHECK_EQ(blob_names.size(), dataset_names.size()) <<
		" the number of blob names and dataset names must be equal";
	size_t num_features = blob_names.size();

	for (size_t i = 0; i < num_features; i++) {
		CHECK(feature_extraction_net->has_blob(blob_names[i]))
			<< "Unknown feature blob name " << blob_names[i]
			<< " in the network " << feature_extraction_proto;
	}

	int num_mini_batches = atoi(argv[++arg_pos]);

	std::vector<boost::shared_ptr<db::DB> > feature_dbs;
	std::vector<boost::shared_ptr<db::Transaction> > txns;
	std::vector<FILE*> fout_feature;
	vector<int> num_samples(num_features, 0);
	for (size_t i = 0; i < num_features; ++i) {
		LOG(INFO) << "Opening dataset " << dataset_names[i];
		FILE* fout = fopen(dataset_names[i].c_str(), "wb");
		fout_feature.push_back(fout);
		//		fwrite(&(num_samples[i]), sizeof(int), 1, fout);
		fwrite(&num_mini_batches, sizeof(int), 1, fout);
	}

	LOG(ERROR) << "Extacting Features";

	Datum datum;
	std::vector<int> image_indices(num_features, 0);
	for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
		feature_extraction_net->Forward();
		for (int i = 0; i < num_features; ++i) {
			const boost::shared_ptr<Blob<Dtype> > feature_blob =
				feature_extraction_net->blob_by_name(blob_names[i]);

			int num_axes = feature_blob->num_axes();
			fwrite(&num_axes, sizeof(int), 1, fout_feature[i]);
			int dim_features = 1;
			for (int idx = 0; idx < num_axes; ++idx) {
				int shape_i = feature_blob->shape(idx);
				fwrite(&shape_i, sizeof(int), 1, fout_feature[i]);
				dim_features *= shape_i;
			}
			fwrite(feature_blob->cpu_data(), sizeof(Dtype), dim_features, fout_feature[i]);
		}  // for (int i = 0; i < num_features; ++i)
	}  // for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)
	// write the last batch
	for (int i = 0; i < num_features; ++i)
		fclose(fout_feature[i]);

	LOG(ERROR) << "Successfully extracted the features!";
	return 0;
}
