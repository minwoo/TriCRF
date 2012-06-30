/*
 * Copyright (C) 2010 Minwoo Jeong (minwoo.j@gmail.com).
 * This file is part of the "TriCRF" distribution.
 * http://github.com/minwoo/TriCRF/
 * This software is provided under the terms of Modified BSD license: see LICENSE for the detail.
 */

// max header
#include "Evaluator.h"
#include "Utility.h"

// stl header
#include <cassert>
#include <cfloat>
#include <cmath>
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <numeric>

using namespace std;

namespace tricrf {

/** Constructor.
*/
Evaluator::Evaluator() {
	initialize();
}

/** Constructor.
*/
Evaluator::Evaluator(Parameter& param, bool bio) {
	initialize();
	encode(param, bio);
}

/** Initialize evaluator.
*/
void Evaluator::initialize() {
	/// initialize
	n_correct = 0;
	n_event = 0;
	n_sequence = 0;
	n_class = class_vec.size();
	accuracy = 0.0;
	loglikelihood = 0.0;
	nGuessPhrase_ = 0;
	nCorrectPhrase_ = 0;
	nTruePhrase_ = 0;	
	
	/// F1 score
	if (n_class > 0) {
		true_class.resize(n_class, 0);
		fill(true_class.begin(), true_class.end(), 0);
		guess_class.resize(n_class, 0);
		fill(guess_class.begin(), guess_class.end(), 0);
		correct_class.resize(n_class, 0);
		fill(correct_class.begin(), correct_class.end(), 0);
		per_class_prec.resize(n_class, 0.0);
		fill(per_class_prec.begin(), per_class_prec.end(), 0.0);
		per_class_rec.resize(n_class, 0.0);
		fill(per_class_rec.begin(), per_class_rec.end(), 0.0);
		per_class_f1.resize(n_class, 0.0);
		fill(per_class_f1.begin(), per_class_f1.end(), 0.0);
	}
}

/** Encode the class information
*/
void Evaluator::encode(Parameter& param, bool bio) {
	map<string, size_t> m_StateMap = param.getState().first;
	vector<string> m_StateVec = param.getState().second;

	if (!bio) {	/// does not use BIO encoding scheme
		class_map = m_StateMap;
		class_vec = m_StateVec;
		for (size_t i = 0; i < class_vec.size(); ++i) {
			bio_index.push_back(i);
			is_begin.insert(make_pair(i, 1));		
		}
		
		is_bio_encoding = false;
	} else {	 /// use BIO encoding scheme
		vector<string>::iterator it = m_StateVec.begin();
		for (; it != m_StateVec.end(); ++it) {
			vector<string> tok = tokenize(*it, "-");
			if (tok.size() > 1 && (tok[0] == "B" || tok[0] == "I")) {
				if (class_map.find(tok[1]) == class_map.end()) {
					class_map.insert(make_pair(tok[1], class_vec.size()));
					bio_index.push_back(class_vec.size());
					class_vec.push_back(tok[1]);
				} else 
					bio_index.push_back(class_map[tok[1]]); 
				if (tok[0] == "B")
					is_begin.insert(make_pair(m_StateMap[*it], 1));
			} else {
				class_map.insert(make_pair(*it, class_vec.size()));
				bio_index.push_back(class_vec.size());
				class_vec.push_back(*it);
				is_begin.insert(make_pair(m_StateMap[*it], 1));				
			}
		} // for
		is_bio_encoding = true;
	}	// if else
	
	assert(bio_index.size() == m_StateVec.size());
	
	outside_class = m_StateMap["O"];

	/** Out of class */
	class_map.insert(make_pair("!OUT_OF_CLASS!", class_vec.size()));
	class_vec.push_back("!OUT_OF_CLASS!");
	OUT_OF_CLASS = class_vec.size() - 1;
}

/** Chunking the sequence.
	Using BIO encoding, this function does chunking for a given sequence.
	@return chunk phrase
*/
vector<pair<size_t, pair<size_t, size_t> > > Evaluator::chunk(vector<size_t> seq) {
	vector<pair<size_t, pair<size_t, size_t> > > phrase;	///< return

	size_t label, spos, epos;
	bool isinphrase = false, isempty = true;
	
	for (size_t i = 0; i < seq.size(); i++) {
		if (seq[i] == outside_class) {	/// outside class
			if (!isempty) 
				phrase.push_back(make_pair(label, make_pair(spos, epos)));
			isempty = true;
			isinphrase = false;
		} else if (is_begin.find(seq[i]) != is_begin.end()) {	/// B-X class
			if (!isempty) 
				phrase.push_back(make_pair(label, make_pair(spos, epos)));
			label = bio_index[seq[i]];
			spos = i; epos = i;
			isempty = false;
			isinphrase = true;
		} else if (isinphrase) {	/// I-X class
			size_t label2 = 10000;
			if (bio_index.size() < seq[i])
				label2 = bio_index[seq[i]];		
			if (label != label2) {	 /// but ...
				if (!isempty) 
					phrase.push_back(make_pair(label, make_pair(spos, epos)));
				if (bio_index.size() < seq[i])
					label = bio_index[seq[i]];
				else
					label = OUT_OF_CLASS;
				spos = i; epos = i;
				isempty = false;
			} else 
				epos = i;
		} else {
			if (!isempty) 
				phrase.push_back(make_pair(label, make_pair(spos, epos)));
			if (bio_index.size() < seq[i])
				label = bio_index[seq[i]];
			else
				label = OUT_OF_CLASS;
			spos = i; epos = i;
			isempty = false;
			isinphrase = true;
		}
	}
	if (!isempty)
		phrase.push_back(make_pair(label, make_pair(spos, epos)));

	return phrase;
}

size_t Evaluator::append(Parameter& param, vector<string> ref, vector<string> hyp) {
	vector<size_t> ref_d, hyp_d;
	map<string, size_t> m_StateMap = param.getState().first;
	vector<string> m_StateVec = param.getState().second;

	for (size_t i = 0; i < ref.size(); i++) {
		if (m_StateMap.find(ref[i]) != m_StateMap.end()) 
			ref_d.push_back(m_StateMap[ref[i]]);
		else
			ref_d.push_back(m_StateVec.size());
		if (m_StateMap.find(hyp[i]) != m_StateMap.end()) 
			hyp_d.push_back(m_StateMap[hyp[i]]);
		else
			hyp_d.push_back(m_StateVec.size());
	}
	return append(ref_d, hyp_d);
}

/** Append the reference and hypothesis.
*/
size_t Evaluator::append(vector<size_t> ref, vector<size_t> hyp) {
	assert(ref.size() == hyp.size());

	// accuracy
	for (size_t i = 0; i < ref.size(); i++) {
		if (ref[i] == hyp[i]) 
			n_correct ++;
		n_event ++;
	}
	n_sequence ++;
	
	/// f1-score
	size_t g_index, t_index;
	if (is_bio_encoding) { /// bio enconding
		vector<pair<size_t, pair<size_t, size_t> > > ref_class, hyp_class;
		ref_class = chunk(ref);
		hyp_class = chunk(hyp);
		/// for reference
		vector<pair<size_t, pair<size_t, size_t> > >::iterator rit = ref_class.begin();
		for (; rit != ref_class.end(); ++rit ) {
			true_class[rit->first] ++;
			nTruePhrase_ ++;			
		}
		/// for hypothesis
		vector<pair<size_t, pair<size_t, size_t> > >::iterator hit = hyp_class.begin();
		for (; hit != hyp_class.end(); ++hit ) {
			guess_class[hit->first] ++;
			nGuessPhrase_ ++;			
		}
		/// correct 
		rit = ref_class.begin();
		hit = hyp_class.begin();
		for (; rit != ref_class.end() && hit != hyp_class.end(); ) {
			g_index = hit->first;
			t_index = rit->first;
			if (rit->second.first == hit->second.first && rit->second.second == hit->second.second) {
				if (g_index == t_index) { 
					correct_class[t_index] ++;
					nCorrectPhrase_ ++;					
				}
				++rit; ++hit;
			} else if (rit->second.second < hit->second.first) {
				++rit;
			} else if (rit->second.first > hit->second.second) {
				++hit;
			} else {
				++rit; ++hit;
			}		
		}
	} else { ///< no bio encoding
		for (size_t i = 0; i < ref.size(); i++) {
			g_index = hyp[i];
			t_index = ref[i];
			guess_class[g_index] ++;
			true_class[t_index] ++;
			nGuessPhrase_ ++;
			nTruePhrase_ ++;							
						
			if (g_index == t_index) {
				correct_class[t_index] ++;
				nCorrectPhrase_ ++;					
			}
		}
	}
	
	return n_sequence;
}

/** Calculate the F1.
*/
void Evaluator::calculateF1() {
	/// per-class f1-scores
	size_t num_data = 0;
	micro_prec = 0.0;
	micro_rec = 0.0;
	micro_f1 = 0.0;
	macro_prec = 0.0;
	macro_rec = 0.0;
	macro_f1 = 0.0;
	for (size_t i = 0; i < class_vec.size(); i++) {
		//if (class_vec[i] == "O" || i == OUT_OF_CLASS) 
		//	continue;
		if (guess_class[i] == 0 || correct_class[i] == 0)
			per_class_prec[i] = 0.0;
		else 
			per_class_prec[i] = (double)correct_class[i] * 100.0 / guess_class[i];
		if (true_class[i] == 0 || correct_class[i] == 0)
			per_class_rec[i] = 0.0;
		else
			per_class_rec[i] = (double)correct_class[i] * 100.0 / true_class[i];
		if ((per_class_prec[i] + per_class_rec[i]) == 0.0)
			per_class_f1[i] = 0.0;
		else
			per_class_f1[i] = 2.0 * (per_class_prec[i] * per_class_rec[i]) / (per_class_prec[i] + per_class_rec[i]);
		micro_prec += per_class_prec[i] * true_class[i];
		micro_rec += per_class_rec[i] * true_class[i];
		num_data += true_class[i];
	}
	macro_prec = accumulate(per_class_prec.begin(), per_class_prec.end(), 0.0) / n_class;
	macro_rec = accumulate(per_class_rec.begin(), per_class_rec.end(), 0.0) / n_class;
	if ((macro_prec + macro_rec) != 0.0)
		macro_f1 = 2.0 * (macro_prec * macro_rec) / (macro_prec + macro_rec);
	
	micro_prec /= num_data;
	micro_rec /= num_data;
	/*if (nGuessPhrase_ > 0)
		micro_prec = 100.0 * (double)nCorrectPhrase_ / (double)nGuessPhrase_;
	if (nTruePhrase_ > 0)
		micro_rec = 100.0 * (double)nCorrectPhrase_ / (double)nTruePhrase_;*/
	if ((micro_prec + micro_rec) != 0.0)
		micro_f1 = 2.0 * (micro_prec * micro_rec) / (micro_prec + micro_rec);
}

/** Add likelihood.
	@return loglikelihood
*/
double Evaluator::addLikelihood(double p) {
	double t;
	if (p == 0.0)
		t = LOG_ZERO;
	else
		t = (double)log(p);
	if (finite(t))
		loglikelihood -= t;
	else
		loglikelihood -= LOG_ZERO;

	return loglikelihood;
}

double Evaluator::addLikelihood(double p, double w) {
	double t;
	if (p == 0.0)
		t = LOG_ZERO;
	else
		t = (double)log(p);
	if (finite(t))
		loglikelihood -= w * t;
	else
		loglikelihood -= w * LOG_ZERO;

	return loglikelihood;
}

/** Substract loglikelihood.
	@return loglikelihood
*/
double Evaluator::subLoglikelihood(double p) {
	loglikelihood += p;
}

/** Get loglikelihood.
	@return loglikelihood
*/
double Evaluator::getObjFunc() {
	return loglikelihood;
}

/** Get loglikelihood.
	@return loglikelihood
*/
double Evaluator::getLoglikelihood() {
	//return -loglikelihood / n_event; // normalized
	return -loglikelihood;
}

/** Get accuracy.
	@return accuracy
*/
double Evaluator::getAccuracy() {
	/// accuracy
	accuracy = n_correct / (double)n_event * 100.0;
	return accuracy;
}

/** Get macro F1.
	@return precision, recall, and f1
*/
vector<double> Evaluator::getMacroF1() {
	vector<double> ret;	 ///< return vector
	ret.push_back(macro_prec);
	ret.push_back(macro_rec);
	ret.push_back(macro_f1);
	return ret;
}

/** Get micro F1.
	@return precision, recall, and f1
*/
vector<double> Evaluator::getMicroF1() {
	vector<double> ret;	 ///< return vector
	ret.push_back(micro_prec);
	ret.push_back(micro_rec);
	ret.push_back(micro_f1);
	return ret;
}

size_t Evaluator::sizeClass() {
	return n_class;
}

void Evaluator::Print(Logger *logger) {
	logger->report("Accuracy: %6.2f%%: prec: %6.2f%%; rec: %6.2f%%; F1: %6.2f\n", 
		getAccuracy(), micro_prec, micro_rec, micro_f1);
	for (size_t i = 0; i < n_class; i++) {
		if (class_vec[i] != "O" && class_vec[i] != "!OUT_OF_CLASS!")
			logger->report("%17s: prec: %6.2f%%; rec: %6.2f%%; F1: %6.2f\n", 
				class_vec[i].c_str(), per_class_prec[i], per_class_rec[i], per_class_f1[i]);
	}
}


}	// namespace tricrf

