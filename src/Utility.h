/*
 * Copyright (C) 2010 Minwoo Jeong (minwoo.j@gmail.com).
 * This file is part of the "TriCRF" distribution.
 * http://github.com/minwoo/TriCRF/
 * This software is provided under the terms of Modified BSD license: see LICENSE for the detail.
 */

#ifndef __UTIL_H__
#define __UTIL_H__

/// standard headers
#include <string>
#include <vector>
#include <map>
#include <math.h>
#include <ctime>
#include <cmath>
#include <cfloat>
#include <fstream>
#include <stdarg.h>
#include <limits>

namespace tricrf {

#define MAX_HEADER "===============================================\n  TriCRF - Triangular-chain CRF\n===============================================\n"

/// tokenizer
std::vector<std::string> tokenize(const std::string& str, const std::string& delimiters = " \t");

/// Logger
class Logger {
private:
	size_t m_Level;
	FILE *m_File;
	std::string getTime();
public:
	Logger();
	Logger(const std::string& filename, size_t level = 1);
	~Logger();
	void setLevel(size_t level);
	int report(size_t level, const char *fmt, ...);
	int report(const char *fmt, ...);
};

/* Configurator class.
	@class Configurator
*/
class Configurator {
private:
	std::string m_filename;
	std::map<std::string, std::vector<std::string> > config;
public:
	Configurator();
	Configurator(const std::string& filename);
	bool parse(const std::string& filename);
	std::string getFileName();
	bool isValid(const std::string& key);
	std::string get(const std::string& key);
	std::vector<std::string> gets(const std::string& key);
};

//  boost timer  -------------------------------------------------------------------//
//  Copyright Beman Dawes 1994-99.
//  See accompanying license for terms and conditions of use.
//  See http://www.boost.org/libs/timer for documentation.
class timer {
 public:
	timer() { _start_time = std::clock(); } 
	void   restart() { _start_time = std::clock(); } 
	double elapsed() const { return  double(std::clock() - _start_time) / CLOCKS_PER_SEC; }
	double elapsed_max() const { return (double(std::numeric_limits<std::clock_t>::max())	- double(_start_time)) / double(CLOCKS_PER_SEC); 	}
	double elapsed_min() const  { return double(1)/double(CLOCKS_PER_SEC); }
private:
	std::clock_t _start_time;
}; // timer

/// finite testing function
#if defined(_MSC_VER) || defined(__BORLANDC__)
inline int finite(double x) { return _finite(x); }
#endif

/// log zero
const double LOG_ZERO = log(DBL_MIN);

} // namespace tricrf

#endif
