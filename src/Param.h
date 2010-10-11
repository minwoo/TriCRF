/*
 * Copyright (C) 2010 Minwoo Jeong (minwoo.j@gmail.com).
 * This file is part of the "TriCRF" distribution.
 * http://github.com/minwoo/TriCRF/
 * This software is provided under the terms of LGPL.
 */

#ifndef __PARAM_H__
#define __PARAM_H__

/// max headers
#include "Utility.h"
/// standard headers
#include <vector>
#include <string>
#include <map>

namespace tricrf {

/** Structure for Observation Parameter.
*/
struct ObsParam {
	size_t y, fid;
	double fval;
};

/** Structure for State Parameter.
*/
struct StateParam {
	size_t y1, y2, fid;
	double fval;
};

/** Typedef for Map.
*/
typedef std::map<std::string, size_t> Map;

/** Typedef for Vec.
*/
typedef std::vector<std::string> Vec;


/** Parameter class.
	@class Parameter
*/
class Parameter {
protected:
	/// Weight
	size_t n_weight;
	std::vector<double> m_Weight;
	std::vector<double> m_Gradient;
	std::vector<double> m_Count;
	
	/// Dictionary
	Map m_FeatureMap;
	Vec m_FeatureVec;
	Map m_StateMap;
	Vec m_StateVec;
	

	/// Options
	std::string mEDGE;
	size_t m_default_oid;

public:
	/// 
	//std::vector<size_t> m_StateID;

	Parameter();
	~Parameter();

	/// Parameter index
	std::vector<std::vector<std::pair<size_t, size_t> > > m_ParamIndex;

	/// weight vector
	void initialize();
	void initializeGradient();
	void initializeGradient2();
	size_t size(); 
	void clear(bool state = false);

	/// Parameters 
	double* getWeight();
	double* getGradient();
	void setWeight(double* theta);

	std::vector<StateParam> m_StateIndex;
	std::vector<ObsParam> makeObsIndex(std::vector<std::pair<size_t, double> >& obs);
	std::vector<ObsParam> makeObsIndex(std::vector<std::pair<size_t, double> >& obs, std::map<size_t, size_t>& beam);
	std::vector<ObsParam> makeObsIndex(std::vector<std::pair<std::string, double> >& obs);
	int findObs(const std::string& key);
	int findState(const std::string& key);
	size_t getDefaultState();

	/// Dictionary access functions
	size_t sizeFeatureVec();
	size_t sizeStateVec();
	std::pair<Map, Vec> getState();
	//int findState(size_t key); 

	/// Update and test the parameters
	size_t addNewState(const std::string& key);
	size_t addNewObs(const std::string& key);
	size_t updateParam(size_t oid, size_t pid,  double fval = 1.0);
	void endUpdate();
	void makeStateIndex(bool makeIndex = true);
	std::vector<StateParam> makeStateIndex(size_t y1);
	void makeActiveIndex(double eta = 1E-02);

	// for tied potential
	std::vector<StateParam> m_SelectedStateIndex;
	std::vector<StateParam> m_RemainStateIndex;
	void makeTiedPotential(double K);
	std::vector<size_t> remain_fid;
	std::vector<double> remain_count;
	std::vector<std::vector<size_t> > m_SelectedStateList1;
	std::vector<std::vector<size_t> > m_SelectedStateList2;

	/// save and load
	bool save(std::ofstream& f);
	bool load(std::ifstream& f);

	/// Reporting
	void print(Logger *log);
};

} // namespace tricrf

#endif

