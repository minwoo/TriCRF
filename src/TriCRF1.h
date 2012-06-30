/*
 * Copyright (C) 2010 Minwoo Jeong (minwoo.j@gmail.com).
 * This file is part of the "TriCRF" distribution.
 * http://github.com/minwoo/TriCRF/
 * This software is provided under the terms of Modified BSD license: see LICENSE for the detail.
 */

#ifndef __TRICRF1_H__
#define __TRICRF1_H__

/// max headers
#include "CRF.h"
/// standard headers
#include <vector>
#include <string>
#include <map>

namespace tricrf {

/** Triangular-chain Conditional Random Fields (Model1).
	@class TriCRF1
*/
class TriCRF1 : public CRF {
protected:
	/// Data sets
	Data<TriStringSequence> m_TrainSet;	 ///< Train data
	Data<TriStringSequence> m_DevSet;	///< Development data (held-out data)
	std::vector<std::vector<TriSequence> > m_TrainLabelSet;
	
	std::vector<std::vector<long double> > m_M;			///< M matrix ; edge transition 
	std::vector<std::vector<long double> > m_R;			///< R matrix ; node observation
	std::vector<std::vector<long double> > m_Alpha;	///< Alpha matrix
	std::vector<std::vector<long double> > m_Beta;		///< Beta matrix
	std::vector<long double> m_Gamma;			///< Gamma matrix ; topic prior
	std::vector<long double> m_Z;			///< Z matrix ; topic prior	

	/// Parameters
	std::vector<Parameter> m_ParamSeq;
	Parameter m_ParamTopic;
	std::map<std::pair<size_t, size_t>, size_t> m_Mapping;
	std::map<std::pair<size_t, size_t>, size_t> m_RMapping;

	/// Variables for computation
	size_t m_topic_size;
	std::vector<size_t> m_state_size;	 ///< re-definition
	size_t m_state_size2;

	/// Inference
	void calculateFactors(TriStringSequence &seq);	///< Calculating the factors
	void calculateEdge();
	void forward();	 ///< Forward recursion
	void backward();	///< Backward recursion
	long double getPartitionZ();	///< Z
	long double calculateProb(TriStringSequence& seq);	///< Prob(y|x)
	std::vector<size_t> viterbiSearch(size_t& max_z, long double& prob);	///< Find the best path

	/// Parameter Estimation
	bool estimateWithLBFGS(size_t max_iter, double sigma, bool L1 = false, double eta = 1E-05);
	bool estimateWithPL(size_t max_iter, double sigma, bool L1 = false, double eta = 1E-05);

public:
	TriCRF1();
	TriCRF1(Logger *logger);
	
	/// Data manipulation
	void readTrainData(const std::string& filename);
	void readDevData(const std::string& filename);

	/// Model 
	bool loadModel(const std::string& filename);
	bool saveModel(const std::string& filename);

	/// Training 
	void clear();
	void initializeModel();
	bool pretrain(size_t max_iter = 100, double sigma = 20, bool L1 = false); 
	bool train(size_t max_iter = 100, double sigma = 20, bool L1 = false); 

	/// Testing
	bool test(const std::string& filename, const std::string& outputfile = "", bool confidence = false);	
	
	Parameter& getTopicParam() { return m_ParamTopic; };
	std::vector<Parameter>& getSeqParam() { return m_ParamSeq; };

};	///< TriCRF1


} // namespace tricrf

#endif
