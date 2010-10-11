/*
 * Copyright (C) 2010 Minwoo Jeong (minwoo.j@gmail.com).
 * This file is part of the "TriCRF" distribution.
 * http://github.com/minwoo/TriCRF/
 * This software is provided under the terms of LGPL.
 */

/// max headers
#include "MaxEnt.h"
#include "CRF.h"
#include "TriCRF1.h"
#include "TriCRF2.h"
#include "TriCRF3.h"
/// standard headers
#include <cassert>
#include <cfloat>
#include <cmath>
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <iostream>

using namespace std;

int main(int argc, void** argv) {
	////////////////////////////////////////////////////////////////
	///	 Model
	////////////////////////////////////////////////////////////////
	tricrf::MaxEnt *model;

	////////////////////////////////////////////////////////////////
	///	 Parameters
	////////////////////////////////////////////////////////////////
	vector<string> model_file, train_file, dev_file, test_file, output_file;
	string initialize_method, estimation_method;
	size_t max_iter, init_iter;
	double l1_prior, l2_prior;
	enum {MaxEnt = 0, CRF, TriCRF1, TriCRF2, TriCRF3} model_type;
	bool train_mode = false, testing_mode = false;
	bool confidence = false;

	////////////////////////////////////////////////////////////////
	///	 Reading the configuration file
	////////////////////////////////////////////////////////////////
	char config_filename[128];
	if (argc > 1) {
		strcpy(config_filename, (char*)argv[1]);
	}
	else {
		cout << MAX_HEADER;
		cout << "[Usage] max config_file \n\n";
		exit(1);		
	}
	tricrf::Configurator config(config_filename);

	////////////////////////////////////////////////////////////////
	///	 Logger
	////////////////////////////////////////////////////////////////
	tricrf::Logger *log = NULL;
	if (config.isValid("log_file")) {
		size_t log_mode = 2;
		if (config.isValid("log_mode"))
			log_mode = atoi(config.get("log_mode").c_str());
		log = new tricrf::Logger(config.get("log_file"), log_mode);
		log->report("[Configurating]\n");
		log->report(" Configuration File = %s\n\n", config.getFileName().data());
	}
	
	////////////////////////////////////////////////////////////////
	///	 Selecting the model
	////////////////////////////////////////////////////////////////
	if (config.isValid("model_type")) {
		string type_str = config.get("model_type");
		if (type_str == "MaxEnt" || type_str == "maxent") {
			model_type = MaxEnt;
			if (log != NULL)
				model = new tricrf::MaxEnt(log);
			else
				model = new tricrf::MaxEnt();
		} else if (type_str == "TriCRF1" || type_str == "tricrf1") {
			model_type = TriCRF1;
			if (log != NULL)
				model = new tricrf::TriCRF1(log);
			else
				model = new tricrf::TriCRF1();
		} else if (type_str == "TriCRF2" || type_str == "tricrf2") {
			model_type = TriCRF2;
			if (log != NULL)
				model = new tricrf::TriCRF2(log);
			else
				model = new tricrf::TriCRF2();
		} else if (type_str == "TriCRF3" || type_str == "tricrf3") {
			model_type = TriCRF3;
			if (log != NULL)
				model = new tricrf::TriCRF3(log);
			else
				model = new tricrf::TriCRF3();
		} else if (type_str == "CRF" || type_str == "crf") {
			model_type = CRF;
			if (log != NULL)
				model = new tricrf::CRF(log);
			else
				model = new tricrf::CRF();
		} else {
			cerr << "Unspecified model type\n";
			exit(1);
		}
	}

	////////////////////////////////////////////////////////////////
	///	 Mode
	////////////////////////////////////////////////////////////////
	if (config.isValid("mode")) 
		train_mode = (config.get("mode") == "train" || config.get("mode") == "both" ? true : false);
	if (config.isValid("mode")) 
		testing_mode = (config.get("mode") == "test" || config.get("mode") == "both" ? true : false);

	////////////////////////////////////////////////////////////////
	///	 Data Files
	////////////////////////////////////////////////////////////////
	if (config.isValid("train_file"))
		train_file = config.gets("train_file");
	if (config.isValid("dev_file"))
		dev_file = config.gets("dev_file");
	if (config.isValid("test_file"))
		test_file = config.gets("test_file");

	////////////////////////////////////////////////////////////////
	///	 Model File
	////////////////////////////////////////////////////////////////
	if (config.isValid("model_file")) {
		model_file = config.gets("model_file");
	}

	////////////////////////////////////////////////////////////////
	///	 Pruning
	////////////////////////////////////////////////////////////////
	if (config.isValid("prune")) {
		double prune = atof(config.get("prune").c_str());
		model->setPrune(prune);
	}
	else
		model->setPrune(1000);

	////////////////////////////////////////////////////////////////
	///	 Training mode
	////////////////////////////////////////////////////////////////
	if (train_mode) { 
		assert(train_file.size() == model_file.size());
		if (train_file.size() <= 0) {
			cerr << "Invalid setting. Please see the configuration\n";
			return -1;
		}
		
		for (size_t iter = 0; iter < train_file.size(); iter++) {
			log->report("\n\nTraining File = %s\n\n", train_file[iter].data());
			model->clear();
			model->readTrainData(train_file[iter]);
			model->initializeModel();	// initialize the model
			if (dev_file.size() != 0) {
				assert(train_file.size() == dev_file.size());
				model->readDevData(dev_file[iter]);
			}
			if (initialize_method == "")
				model->initializeModel();
			
			if (config.isValid("iter"))
				max_iter = atoi(config.get("iter").c_str());
			else
				max_iter = 100;
			
			// initializing the parameter
			bool init_param = false;
			if (config.isValid("initialize")) {
				if (config.get("initialize") == "PL") {
					if (config.isValid("initialize_iter"))
						init_iter = atoi(config.get("initialize_iter").c_str());
					else
						init_iter = 30;
				}
				init_param = true;
			}

			string type_str = "LBFGS-L2";	///< default estimation method
			if (config.isValid("estimation")) {
				type_str = config.get("estimation");
			}

			if (type_str == "LBFGS-L1") {
				/// LBFGS-L1
				if (config.isValid("l1_prior"))
					l1_prior = atof(config.get("l1_prior").c_str());
				else
					l1_prior = 0.0;
				
				if (init_param) {
					if (!model->pretrain(max_iter, l1_prior, true)) {
						cerr << "PL training terminates with error. anyway, we will go.\n\n";
						//return -1;
					}
				}
				if (!model->train(max_iter, l1_prior, true)) {
					cerr << "training terminates with error\n\n";
					return -1;
				}		
			} else { 
				/// LBFGS-L2
				if (config.isValid("l2_prior"))
					l2_prior = atof(config.get("l2_prior").c_str());
				else
					l2_prior = 0.0;

				if (init_param) {
					if (!model->pretrain(max_iter, l2_prior, true)) {
						cerr << "PL training terminates with error. anyway, we will go.\n\n";
						//return -1;
					}
				}
				if (!model->train(max_iter, l2_prior, false)) {
					cerr << "training terminates with error\n\n";
					return -1;
				}		
			}

			if (config.isValid("model_file")) {
				model->saveModel(model_file[iter]);
			}

		} // iteration

	} 
	////////////////////////////////////////////////////////////////
	///	 Testing mode
	////////////////////////////////////////////////////////////////	
	if (testing_mode) { 
		assert(test_file.size() == model_file.size());
		if (model_file.size() == 0 || test_file.size() == 0) {
			cerr << "Invalid setting. Please see the configuration\n";
			return -1;
		}
		if (config.isValid("output_file")) {
			output_file = config.gets("output_file");
			assert(test_file.size() == output_file.size());
			if (config.isValid("confidence"))
				confidence = (config.get("confidence") == "true" ? true : false);
		}

		for (size_t iter = 0; iter < test_file.size(); iter++) {
			log->report("\n\nTest File = %s\n\n", test_file[iter].data());
			model->clear();
			if (!model->loadModel(model_file[iter])) {
				cerr << "Model loading error\n";
				return -1;
			}
			if (config.isValid("output_file")) {
				model->test(test_file[iter], output_file[iter], confidence);
			} else
				model->test(test_file[iter]);
		}
	}

}
