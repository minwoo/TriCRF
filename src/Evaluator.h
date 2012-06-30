/*
 * Copyright (C) 2010 Minwoo Jeong (minwoo.j@gmail.com).
 * This file is part of the "TriCRF" distribution.
 * http://github.com/minwoo/TriCRF/
 * This software is provided under the terms of Modified BSD license: see LICENSE for the detail.
 */

#ifndef __EVALUATOR_H__
#define __EVALUATOR_H__

/// max headers
#include "Param.h"
#include "Data.h"
#include "Utility.h"
/// standard headers
#include <vector>
#include <string>
#include <map>

namespace tricrf {

/** Evaluator class.
	@class Evaluator
*/
class Evaluator {
private:
	/// Encoder
	std::map<std::string, size_t> class_map;
	std::vector<std::string> class_vec;
	std::vector<size_t> bio_index;
	std::map<size_t, size_t> is_begin;
	size_t outside_class;
	bool is_bio_encoding;

	/// f1-scores
	std::vector<size_t> true_class;
	std::vector<size_t> guess_class;
	std::vector<size_t> correct_class;
	std::vector<double> per_class_prec;
	std::vector<double> per_class_rec;
	std::vector<double> per_class_f1;

	/// basic information
	size_t n_correct;			/// number of correct events
	size_t n_event;			/// total number of events
	size_t n_sequence;		/// total number of sequence
	size_t n_class;
	size_t OUT_OF_CLASS;

	/// masures
	double loglikelihood;	/// log-likelihood
	double accuracy;			/// accuracy
	double macro_f1;		/// macro averaged F1 score
	double macro_prec;	/// macro averaged precision
	double macro_rec;		/// macro averaged recall
	double micro_f1;			/// micro averaged F1 score
	double micro_prec;		/// micro averaged precision
	double micro_rec;		/// micro averaged recall
	size_t nTruePhrase_, nGuessPhrase_, nCorrectPhrase_;	
	
	/// private methods
	std::vector<std::pair<size_t, std::pair<size_t, size_t> > > chunk(std::vector<size_t> seq);

public:
		
	/// basic methods
	Evaluator();
	Evaluator(Parameter& param, bool bio = true);
	void initialize();	

	/// encoding and calculating
	void encode(Parameter& param, bool bio = true);
	size_t append(Parameter& param, std::vector<std::string> ref, std::vector<std::string> hyp);
	size_t append(std::vector<size_t> ref, std::vector<size_t> hyp);
	void calculateF1();

	/// log-likelihood
	double subLoglikelihood(double p);
	double addLikelihood(double p);
	double addLikelihood(double p, double w);
	double getObjFunc();
	double getLoglikelihood();
	size_t sizeClass();

	/// accuracy and f1-scores
	double getAccuracy();
	std::vector<double> getMacroF1();
	std::vector<double> getMicroF1();
	void Print(Logger *logger);	

};


} // namespace tricrf

#endif

