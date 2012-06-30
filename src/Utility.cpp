/*
 * Copyright (C) 2010 Minwoo Jeong (minwoo.j@gmail.com).
 * This file is part of the "TriCRF" distribution.
 * http://github.com/minwoo/TriCRF/
 * This software is provided under the terms of Modified BSD license: see LICENSE for the detail.
 */

/// max headers
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
#include <time.h>
#include <stdio.h>

using namespace std;

namespace tricrf {

/** Tokenizer.
	@param	str	a string to be tokenized
	@param	delimiters	delimeter(s) 
	@return	a string vector
*/
vector<string> tokenize(const string& str, const string& delimiters) {
	vector<string> tokens;
    string::size_type lastPos = str.find_first_not_of(delimiters, 0);
    string::size_type pos  = str.find_first_of(delimiters, lastPos);
    while (string::npos != pos || string::npos != lastPos) {
        tokens.push_back(str.substr(lastPos, pos - lastPos));
        lastPos = str.find_first_not_of(delimiters, pos);
        pos = str.find_first_of(delimiters, lastPos);
    }
	return tokens;
} 

/** Logger.
*/
Logger::Logger() {
	m_File = stderr;
	m_Level = 1;
}

Logger::Logger(const string& filename, size_t level) {
	if (filename == "")
		m_File = stderr;
	else {
		if (!(m_File = fopen(filename.c_str(), "a+"))) 
			throw runtime_error("cannot open data file");
	}

	m_Level = level;
}

Logger::~Logger() {
	if (m_File)
		fclose(m_File);
}

void Logger::setLevel(size_t level) {
	m_Level = level;
}

string Logger::getTime() {
	time_t	unix_time;
	time(&unix_time);
	struct tm	*clock = localtime(&unix_time);

	char tmp_time[1024];
	sprintf(tmp_time, "%04d-%02d-%02d %02d:%02d:%02d", clock->tm_year+1900, clock->tm_mon+1, clock->tm_mday, clock->tm_hour, clock->tm_min, clock->tm_sec);
	return string(tmp_time);
}

int Logger::report(const char *fmt, ...) {
	int ret = 0;

	/// write current time
	if (m_Level > 2) {
		fprintf(m_File, "[%s] ", getTime().c_str());
	}
	
	/// write the message
	if (m_Level > 0) {
		va_list argptr;
		va_start(argptr, fmt);
		ret = vfprintf(m_File, fmt, argptr);
		/// standard out
		if (m_Level > 1) {
			vfprintf(stderr, fmt, argptr);
		}
		va_end(argptr);
	}
	fflush(m_File);

	return ret;
}

int Logger::report(size_t level, const char *fmt, ...) {
	int ret = 0;

	/// write current time
	if (level > 2) {
		fprintf(m_File, "[%s] ", getTime().c_str());
	}
	
	/// write the message
	if (level > 0) {
		va_list argptr;
		va_start(argptr, fmt);
		ret = vfprintf(m_File, fmt, argptr);
		if (level > 1) {
			vfprintf(stderr, fmt, argptr);
		}
		va_end(argptr);
	}
	fflush(m_File);

	return ret;
}

/// Configurator
Configurator::Configurator() {
}

Configurator::Configurator(const string& filename) {
	parse(filename);
}

bool Configurator::parse(const string& filename) {
	/// File stream
	string line;
	ifstream f(filename.c_str());
	if (!f)
		throw runtime_error("cannot open data file");

	/// Initializing
	config.clear();
	
	/// reading the text
	while (getline(f, line)) {
		if (!line.empty() && line[0] != '#') {
			vector<string> tokens = tokenize(line, " =\t");
			if (tokens.size() < 2)
				throw runtime_error("invalid configuration file");
			vector<string> values;
			for (size_t i = 1; i < tokens.size(); i++) {
				if (tokens[i].find("[") != string::npos) {
					vector<string> tok = tokenize(tokens[i], "[-]");
					if (tok.size() < 3)
						throw runtime_error("invalid configuration file");
					for (size_t i = atoi(tok[1].c_str()); i <= atoi(tok[2].c_str()); i++) {
						char temp[64];
						sprintf(temp, "%s%d", tok[0].c_str(), i);
						values.push_back(temp);
					}
				} else
					values.push_back(tokens[i]);			}
			config.insert(make_pair(tokens[0], values));
		}
	}
		
	m_filename = filename;
	return true;
}

string Configurator::getFileName() {
	return m_filename;
}

bool Configurator::isValid(const string& key) {
	if (config.find(key) != config.end())
		return true;
	return false;
}

string Configurator::get(const string& key) {
	if (config.find(key) == config.end())
		return NULL;
	else
		return config[key][0];
}

vector<string> Configurator::gets(const string& key) {
	vector<string> result;
	if (config.find(key) != config.end())
		result = config[key];
	return result;
}


}	// namespace tricrf

