/*
 * Copyright (C) 2010 Minwoo Jeong (minwoo.j@gmail.com).
 * This file is part of the "TriCRF" distribution.
 * http://github.com/minwoo/TriCRF/
 * This software is provided under the terms of LGPL.
 */

/// max header
#include "Param.h"
#include "Utility.h"
/// standard headers
#include <cassert>
#include <cfloat>
#include <cmath>
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <fstream>

using namespace std;

namespace tricrf {

/** Constructor.
*/
Parameter::Parameter() {
	mEDGE = "@";
	clear();
}

/** Deconstructor.
*/
Parameter::~Parameter() {
	/// memory free
	//if (m_Weight) 
	//	delete[] m_Weight;
	//if (m_Gradient)
	//	delete[] m_Gradient;
}

/** Size of weight vector.
	@return	size of weight vector
*/
size_t Parameter::size() {
	return n_weight;
}

/** Clear the memory.
*/
void Parameter::clear(bool state) {
	if (!state) {
		m_StateMap.clear();
		m_StateVec.clear();
	}
	m_FeatureMap.clear();
	m_FeatureVec.clear();
	//m_StateID.clear();
	m_Count.clear();
	m_Weight.clear();
	m_Gradient.clear();
	m_ParamIndex.clear();
	n_weight = 0;
	m_StateIndex.clear();
	m_SelectedStateList1.clear();
	m_SelectedStateList2.clear();
}

/** Initialize the weight vector.
*/
void Parameter::initialize() {
	m_Weight.resize(n_weight);
	fill(m_Weight.begin(), m_Weight.end(), 0.0);
	m_Gradient.resize(n_weight);
	fill(m_Gradient.begin(), m_Gradient.end(), 0.0);
}

/** Initialize the gradient vector.
*/
void Parameter::initializeGradient() {
	for (size_t i=0; i < n_weight; i++) {
		m_Gradient[i] = -m_Count[i];
	}
}
void Parameter::initializeGradient2() {
	for (size_t i=0; i < n_weight; i++) {
		m_Gradient[i] = 0.f;
	}
}

/** Get weight vector pointer.
	@warning	The size of weight vector should be larger than 1.
*/
double* Parameter::getWeight() { 
	return &m_Weight[0]; 
}

void Parameter::setWeight(double* theta) {
	for (size_t i = 0; i < n_weight; i++) {
		m_Weight[i] = *(theta++);
	}
}

/** Get gradient vector pointer.
	@warning	The size of gradient vector should be larger than 1.
*/
double* Parameter::getGradient() {
	return &m_Gradient[0]; 
}

/** Make and return the observation index
	@todo	If the index vector is stored in training set, then the training speed can be (slightly) improved.
*/
vector<ObsParam> Parameter::makeObsIndex(vector<pair<size_t, double> >& obs) {
	vector<ObsParam> obs_param; 
	vector<pair<size_t, double> >::iterator iter = obs.begin();
	for (; iter != obs.end(); iter++) {
		vector<pair<size_t, size_t> >& param = m_ParamIndex[iter->first];
		for (size_t i = 0; i < param.size(); ++i) {
			ObsParam element;
			element.y = param[i].first;
			element.fid = param[i].second;
			element.fval = iter->second;
			obs_param.push_back(element);
		}
	}
	return obs_param;
}

// sparse-FB, 2007-11-08 
vector<ObsParam> Parameter::makeObsIndex(vector<pair<size_t, double> >& obs, map<size_t, size_t>& beam) {
	vector<ObsParam> obs_param; 
	vector<pair<size_t, double> >::iterator iter = obs.begin();
	for (; iter != obs.end(); iter++) {
		vector<pair<size_t, size_t> >& param = m_ParamIndex[iter->first];
		size_t index = 0;
		for (size_t i = 0; i < param.size(); ++i) {
			if (beam.find(param[i].first) == beam.end()) 
				continue;
			ObsParam element;
			element.y = param[i].first;
			element.fid = param[i].second;
			element.fval = iter->second;
			obs_param.push_back(element);
		}
	}
	return obs_param;
}

vector<ObsParam> Parameter::makeObsIndex(vector<pair<string, double> >& obs) {
	int pid;
	vector<ObsParam> obs_param; 
	vector<pair<string, double> >::iterator iter = obs.begin();
	for (; iter != obs.end(); iter++) {
		if ((pid = findObs(iter->first)) >= 0) {
			vector<pair<size_t, size_t> >& param = m_ParamIndex[(size_t)pid];
			for (size_t i = 0; i < param.size(); ++i) {
				ObsParam element;
				element.y = param[i].first;
				element.fid = param[i].second;
				element.fval = iter->second;
				obs_param.push_back(element);
			}
		}
	}
	return obs_param;
}

/**	Return the size of feature vector.
*/
size_t Parameter::sizeFeatureVec() { 
	return m_FeatureVec.size(); 
}

/**	Return the size of state vector.
*/
size_t Parameter::sizeStateVec() { 
	return m_StateVec.size(); 
}

/**	Return the state map and vector.
*/
std::pair<Map, Vec> Parameter::getState() { 
	return make_pair(m_StateMap, m_StateVec); 
}

/**	Return the size of feature vector.
*/
//int Parameter::findState(size_t key) { 
//	if (key < m_StateID.size())
//		return m_StateID[key]; 
//	else 
//		return -1;
//}

/**
*/
size_t Parameter::addNewState(const string& key) {
	size_t oid;
	if (m_StateMap.find(key) == m_StateMap.end()) {
		oid = m_StateVec.size();
		m_StateMap.insert(make_pair(key, oid));
		m_StateVec.push_back(key);
	} else {
		oid = m_StateMap[key];
		if (m_StateVec[oid] != key) {
			cerr << "outcome id mismatch error" << endl;
			exit(1);
		}
	}
	return oid;
}

/**
*/
int Parameter::findState(const string& key) {
	int oid = -1;

	if (m_StateMap.find(key) != m_StateMap.end()) {
		oid = m_StateMap[key];
	}

	return oid;
}

/**
*/
int Parameter::findObs(const string& key) {
	int pid = -1;

	if (m_FeatureMap.find(key) != m_FeatureMap.end()) {
		pid = m_FeatureMap[key];
	}

	return pid;
}

/**
*/
size_t Parameter::addNewObs(const string& key) {
	size_t pid;

	if (m_FeatureMap.find(key) != m_FeatureMap.end()) {
		pid = m_FeatureMap[key];
	} else {
		pid = m_FeatureVec.size();
		m_FeatureMap[key] = pid;
		m_FeatureVec.push_back(key);
	}
	return pid;
}

/** Update the parameter.
*/
size_t Parameter::updateParam(size_t oid, size_t pid, double fval) {
	size_t fid;
	assert(m_ParamIndex.size() >= pid);
	if (m_ParamIndex.size() == pid) {	/// New feature
		vector<pair<size_t, size_t> > param;
		fid = n_weight;
		n_weight++;
		m_Count.push_back(fval);
		m_Weight.push_back(0.0);
		m_Gradient.push_back(0.0);
		param.push_back(make_pair(oid, fid));
		m_ParamIndex.push_back(param);
	} else {	 /// A parameter vector is exist 
		vector<pair<size_t, size_t> >& param = m_ParamIndex[pid];
		size_t i;
		for (i = 0; i < param.size(); i++) {
			if (param[i].first == oid) {
				m_Count[param[i].second] += fval;
				break;
			}
		}
		if (i == param.size()) {
			fid = n_weight;
			n_weight++;
			m_Count.push_back(fval);
			m_Weight.push_back(0.0);
			m_Gradient.push_back(0.0);
			param.push_back(make_pair(oid, fid));
	        sort(param.begin(), param.end());
		}
	}
	return n_weight;
}

void Parameter::endUpdate() {
	vector<double> tmp_Count = m_Count;
	fill(m_Count.begin(), m_Count.end(), 0.0);

    size_t fid = 0;
    for (size_t i = 0; i < m_ParamIndex.size(); ++i) {
        vector<pair<size_t, size_t> >& param = m_ParamIndex[i];
        for (size_t j = 0; j < param.size(); ++j) {
			m_Count[fid] = tmp_Count[param[j].second];
            param[j].second = fid;
            fid++;
        }
    }
	assert(fid == n_weight);
}

size_t Parameter::getDefaultState() {
	return m_default_oid;
}

/**
*/
void Parameter::makeStateIndex(bool makeIndex) {
	
	/// Make state pid
    //m_StateID.clear();
	/*
    for (size_t y1=0; y1 < sizeStateVec(); y1++) {
        string fi = mEDGE + m_StateVec[y1];
		if (m_FeatureMap.find(fi) != m_FeatureMap.end()) {
	        size_t pid = m_FeatureMap[fi];
	        m_StateID.push_back(pid);
		}
		else {
			cerr << "Error! no matched state: " << fi << endl;
			exit(0);
		}
    }*/
	//m_default_oid = addNewState("|S|");

	m_SelectedStateList1.resize(sizeStateVec());
	m_SelectedStateList2.resize(sizeStateVec());

	/// Make state index
	m_StateIndex.clear();
	for (size_t y1=0; y1 < sizeStateVec(); y1++) {
		//int pid = findState(y1);
		//if (pid < 0) 
		//	continue;
		string fi = mEDGE + m_StateVec[y1];
		if (m_FeatureMap.find(fi) != m_FeatureMap.end()) {
			size_t pid = m_FeatureMap[fi];
			vector<pair<size_t, size_t> >& param = m_ParamIndex[pid];
			for (size_t i = 0; i < param.size(); i++) {
				StateParam element;
				element.y1 = y1;
				element.y2 = param[i].first;
				element.fid = param[i].second;
				element.fval = 1.0;
				m_StateIndex.push_back(element);
				
				if (makeIndex) {
					// make back pointer (t, t-1)
					vector<size_t> &backpointer = m_SelectedStateList1[element.y2];
					backpointer.push_back(y1);
					vector<size_t> &backpointer2 = m_SelectedStateList2[y1];
					backpointer2.push_back(element.y2);
				}

			}	///< for
		} ///< if else
	} ///< for each state

}

void Parameter::makeActiveIndex(double eta) {
	
	m_SelectedStateList1.clear();
	m_SelectedStateList2.clear();
	m_SelectedStateList1.resize(sizeStateVec());
	m_SelectedStateList2.resize(sizeStateVec());

	/// Make state index
	vector<StateParam>::iterator iter = m_StateIndex.begin();
	for (; iter != m_StateIndex.end(); ++iter) {
		if (abs( exp(m_Weight[iter->fid]) - 1.0 ) > eta) {
			vector<size_t> &backpointer = m_SelectedStateList1[iter->y2];
			backpointer.push_back(iter->y1);
			vector<size_t> &backpointer2 = m_SelectedStateList2[iter->y1];
			backpointer2.push_back(iter->y2);
		}
	}
}

vector<StateParam> Parameter::makeStateIndex(size_t y1) {
	vector<StateParam> state_param; 
	string fi = mEDGE + m_StateVec[y1];
	if (m_FeatureMap.find(fi) != m_FeatureMap.end()) {
		size_t pid = m_FeatureMap[fi];
		vector<pair<size_t, size_t> >& param = m_ParamIndex[pid];
		for (size_t i = 0; i < param.size(); i++) {
			StateParam element;
			element.y1 = i;
			element.y2 = param[i].first;
			element.fid = param[i].second;
			element.fval = 1.0;
			state_param.push_back(element);
		}	///< for
	} ///< if else
	return state_param;
}

/** Make the index for Tied Potential
*/
void Parameter::makeTiedPotential(double K) {
	
	/// Make state index
	m_SelectedStateIndex.clear();
	m_RemainStateIndex.clear();

	remain_count.clear();
	remain_fid.clear();
	vector<double> remain_size;
	for (size_t i = 0; i < sizeStateVec(); i++) {
		size_t temp_fid = updateParam(i, addNewObs("@REMAIN@"), 0.0); // empirical feature count is augmented
		remain_fid.push_back(temp_fid);
		remain_size.push_back(0.0);
		remain_count.push_back(0.0);
	}
	
	// redefinition for tied potential (redundant)
	m_SelectedStateList1.clear();
	m_SelectedStateList2.clear();
	m_SelectedStateList1.resize(sizeStateVec());
	m_SelectedStateList2.resize(sizeStateVec());

	for (size_t y1=0; y1 < sizeStateVec(); y1++) {
		string fi = mEDGE + m_StateVec[y1];
		if (m_FeatureMap.find(fi) != m_FeatureMap.end()) {
			size_t pid = m_FeatureMap[fi];
			vector<pair<size_t, size_t> >& param = m_ParamIndex[pid];
			for (size_t i = 0; i < param.size(); i++) {
				StateParam element;
				element.y1 = y1;
				element.y2 = param[i].first;
				element.fid = param[i].second;
				element.fval = 1.0;
				if (m_Count[element.fid] >= K) {
					m_SelectedStateIndex.push_back(element);

					// make back pointer (t, t-1)
					vector<size_t> &backpointer = m_SelectedStateList1[element.y2];
					backpointer.push_back(y1);
					vector<size_t> &backpointer2 = m_SelectedStateList2[y1];
					backpointer2.push_back(element.y2);

				} else {
					m_RemainStateIndex.push_back(element);
					remain_count[element.y2] += m_Count[element.fid];
					//remain_fid[element.y2] = element.fid;
					remain_size[element.y2] += 1.0;
					m_Count[remain_fid[element.y2]] += m_Count[element.fid]; // empirical feature count is augmented
					m_Count[element.fid] = 0.0;
				}
			}	///< for
		} ///< if else
	} ///< for each state

	/*
	for (size_t j = 0; j < sizeStateVec(); j++) {
			vector<size_t> &backpointer = m_SelectedStateList1[j];
			sort(backpointer.begin(), backpointer.end());
			vector<size_t> &backpointer2 = m_SelectedStateList2[j];
			sort(backpointer2.begin(), backpointer2.end());
	}
	vector<StateParam>::iterator iter = m_RemainStateIndex.begin();
	for (; iter != m_RemainStateIndex.end(); ++iter) {
		m_Count[iter->fid] = remain_count[iter->y2] / remain_size[iter->y2];
	}
	*/
	
	for (size_t i = 0; i < sizeStateVec(); i++) {
		if (remain_size[i] > 0)
			m_Count[remain_fid[i]] /= remain_size[i];
		else
			m_Count[remain_fid[i]] = 0.0;
	}
}

/** Save the model.
	@param	f	output file stream 
	@return	success or failure
*/
bool Parameter::save(ofstream& f) {
	/// Errors	
	if (m_ParamIndex.size() != m_FeatureVec.size())
		return false;

	/// state
	f << "// State ; " << m_StateVec.size() << endl;
    for (size_t i = 0; i < m_StateVec.size(); ++i)
        f << m_StateVec[i] << endl;
	
	/// feature 
    f << "// Feature ; " << m_FeatureVec.size() << endl;
    for (size_t i = 0; i < m_FeatureVec.size(); ++i)
        f << m_FeatureVec[i] << endl;
	
	/// parameter index
    f << "// Parameter ; " << m_ParamIndex.size() << endl;
    for (size_t i = 0; i < m_ParamIndex.size(); ++i) {
        vector<pair<size_t, size_t> >& param = m_ParamIndex[i];
        f << param.size() << ' ';
        for (size_t j = 0; j < param.size(); ++j) {
            f << param[j].first << ' ';
        }
        f << endl;
    }
    /// write the weight vector
    f   << "// Weight ; " << n_weight << endl;
    for (size_t i = 0; i < n_weight; ++i) {
        f << m_Weight[i] << endl;	
	}

	return true;
}

/** Load the model.
	@param	filename	model file name to be loaded
	@return	success or failure
*/
bool Parameter::load(ifstream& f) {
	/// initializing
	clear();
    string line;
	size_t count;

    /// state
	getline(f, line);
	vector<string> tok = tokenize(line);
	if (tok.size() < 4) {
		cerr << "state error\n";
		return false;
	}
	count = atoi(tok[3].c_str());
    for (size_t i = 0; i < count; ++i) {
		getline(f, line);
		m_StateMap[line] = i;
		m_StateVec.push_back(line);
    }

    /// feature
    getline(f, line);
	tok = tokenize(line);
	if (tok.size() < 4) {
		cerr << "feature error\n";
		return false;
	}
	count = atoi(tok[3].c_str());
    for (size_t i = 0; i < count; ++i) {
        getline(f, line);
        m_FeatureMap[line] = i;
        m_FeatureVec.push_back(line);
    }

	/// parameter index
    getline(f, line);
	tok = tokenize(line);
	if (tok.size() < 4)
		return false;
	count = atoi(tok[3].c_str());
    size_t fid = 0;
    vector<pair<size_t, size_t> > param;
    for (size_t i = 0; i < count; ++i) {
        param.clear();
        getline(f, line);
        size_t oid;
        tok = tokenize(line);
		vector<string>::iterator it = tok.begin();	
        ++it; ///< skip count which is only used in binary format
        for (; it != tok.end();) {
            oid = atoi(it->c_str()); ++it;
            param.push_back(make_pair(oid,fid++));
        }
        m_ParamIndex.push_back(param);
    }

	/// weight
    getline(f, line);
	tok = tokenize(line);
	if (tok.size() < 4 || fid != atoi(tok[3].c_str()))
		return false;
	n_weight = fid;
	initialize();
	size_t i = 0;
	for (i = 0; i < fid; i++) {
		getline(f, line);
		assert(!line.empty());
		m_Weight[i] = atof(line.c_str());
	}
	assert(i == n_weight);

	/// setting
	m_Count.resize(n_weight);
	fill(m_Count.begin(), m_Count.end(), 0.0);
	
	return true;
}

/** Print the information.
*/
void Parameter::print(Logger *log) {
	//log->report("[Parameters]\n");
	log->report("  # of States = \t%d\n", m_StateVec.size());
	log->report("  # of Features = \t%d\n", m_FeatureVec.size());
	log->report("  # of Parameters = \t%d\n\n", n_weight);
}

}	// namespace tricrf

