/*
 * Copyright (C) 2010 Minwoo Jeong (minwoo.j@gmail.com).
 * This file is part of the "TriCRF" distribution.
 * http://github.com/minwoo/TriCRF/
 * This software is provided under the terms of LGPL.
 */

#ifndef __MAXENT_H__
#define __MAXENT_H__

/// max headers
#include "Param.h"
#include "Data.h"
/// standard headers
#include <vector>
#include <string>
#include <map>

namespace tricrf {

/** Maximum Entropy Model.
	@class MaxEnt
*/
class MaxEnt {
protected:
	/// Data sets
	Data<Sequence> m_TrainSet;	 ///< Train data
	Data<Sequence> m_DevSet;	///< Development data (held-out data)
	std::vector<double> m_TrainSetCount;	///< Counts for data
	std::vector<double> m_DevSetCount;

	/// Parameter vector
	Parameter m_Param;

	/// Logger 
	Logger *logger;
	
	/// Inference
	virtual std::vector<double> evaluate(Event ev, size_t& max_outcome);

	/// Parameter Estimation
	virtual bool estimateWithLBFGS(size_t max_iter, double sigma, bool L1, double eta = 1E-05);

	/// Prune
	/// for pruning
	std::vector<std::pair<long double, size_t> > m_prune;
	long double m_prune_threshold;


public:
	MaxEnt();	 
	MaxEnt(Logger *logger);
	~MaxEnt();	

	/// Data manipulation
	Event packEvent(std::vector<std::string>& tokens, Parameter* p_Param = NULL, bool test = false);
	Event packEvent2(std::vector<std::string>& tokens, Parameter* p_Param = NULL, bool test = false);
	StringEvent packStringEvent(std::vector<std::string>& tokens, Parameter* p_Param = NULL, bool test = false);
	virtual void readTrainData(const std::string& filename);
	virtual void readDevData(const std::string& filename);
	
	/// Model 
	virtual bool loadModel(const std::string& filename);
	virtual bool saveModel(const std::string& filename);
	virtual bool averageParam() {};

	/// Testing
	virtual bool test(const std::string& filename, const std::string& outputfile = "", bool confidence = false);

	/// Training 
	virtual void clear();
	virtual void initializeModel();
	virtual bool pretrain(size_t max_iter = 100, double sigma = 20, bool L1 = false);
	virtual bool train(size_t max_iter = 100, double sigma = 20, bool L1 = false);

	/// Logger 
	void setLogger(Logger *logger);
	void setPrune(double prune);
	
	Parameter& getParam() { return m_Param; };
};

} // namespace tricrf

#endif
