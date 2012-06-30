/*
 * Copyright (C) 2010 Minwoo Jeong (minwoo.j@gmail.com).
 * This file is part of the "TriCRF" distribution.
 * http://github.com/minwoo/TriCRF/
 * This software is provided under the terms of Modified BSD license: see LICENSE for the detail.
 */

#ifndef __CRF_H__
#define __CRF_H__

/// max headers
#include "MaxEnt.h"
#include "Data.h"
/// standard headers
#include <vector>
#include <string>
#include <map>
#include <valarray>

namespace tricrf {

/** (Linear-chain) Conditional Random Fields.
	@class CRF
*/
class CRF : public MaxEnt {
protected:
	std::vector<long double> m_M;			///< M matrix ; edge transition 
	std::vector<long double> m_M2;			///< M matrix ; edge transition 
	std::vector<long double> m_R;			///< R matrix ; node observation
	std::vector<long double> m_Alpha;	///< Alpha matrix
	std::vector<long double> m_Beta;		///< Beta matrix
	
	/* too slow
	virtual inline size_t MAT3(size_t I, size_t X, size_t Y) {
		return ((m_state_size * m_state_size * (I)) + (m_state_size * (X)) + Y);
	};
	virtual inline size_t MAT2(size_t I, size_t X) {
		return ((m_state_size * (I)) + X);
	};
	*/
	
	/// Variables for computation
	size_t m_state_size;
	size_t m_seq_size;
	size_t m_default_oid;

	/// Inference
	virtual void calculateEdge();	///< Calculating the factors
	virtual void calculateFactors(Sequence &seq);	///< Calculating the factors
	virtual void forward();	 ///< Forward recursion
	virtual void backward();	///< Backward recursion
	virtual long double getPartitionZ();	///< Z
	virtual std::vector<size_t> viterbiSearch(long double& prob);	///< Find the best path

	/// Parameter Estimation
	virtual bool estimateWithLBFGS(size_t max_iter, double sigma, bool L1 = false, double eta = 1E-05);
	virtual bool estimateWithPL(size_t max_iter, double sigma, bool L1 = false, double eta = 1E-05);
	virtual bool averageParam() {};
	
	std::vector<std::vector<size_t> > m_Beam;
	std::vector<std::map<size_t, size_t> > m_BeamMap;
	std::vector<long double> scale;
	std::vector<long double> scale2;
	std::vector<std::vector<size_t> > m_IndexR;
	
public:
	CRF();
	CRF(Logger *logger);
	
	/// Data manipulation
	virtual void readTrainData(const std::string& filename);
	virtual void readDevData(const std::string& filename);

	/// Model 
	virtual bool loadModel(const std::string& filename);
	virtual bool saveModel(const std::string& filename);

	/// Testing
	virtual bool test(const std::string& filename, const std::string& outputfile = "", bool confidence = false);	
	virtual void eval(Sequence seq, std::vector<std::string> &output, long double &prob);
	virtual void eval(Sequence seq, std::vector<std::string> &output, std::vector<long double> &prob);
	virtual void evals(Sequence seq, std::vector<std::string> &output, std::vector<long double> &prob);
		
	/// Training 
	virtual void clear();
	virtual bool pretrain(size_t max_iter = 100, double sigma = 20, bool L1 = false);
	virtual bool train(size_t max_iter = 100, double sigma = 20, bool L1 = false); 
	virtual long double calculateProb(Sequence& seq);	///< Prob(y|x)

};	///< CRF

} // namespace tricrf

#endif
