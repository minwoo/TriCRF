/*
 * Copyright (C) 2010 Minwoo Jeong (minwoo.j@gmail.com).
 * This file is part of the "TriCRF" distribution.
 * http://github.com/minwoo/TriCRF/
 * This software is provided under the terms of Modified BSD license: see LICENSE for the detail.
 */

#ifndef __DATA_H__
#define __DATA_H__

/// standard headers
#include <vector>
#include <string>
#include <map>

namespace tricrf {

/** Event.
	@class Event
*/
struct Event {
	size_t label;
	double fval;
	std::vector<std::pair<size_t, double> > obs;
};

/** String event.
	@class StringEvent
*/
struct StringEvent {
	size_t label;
	double fval;
	std::vector<std::pair<std::string, double > > obs;
};

/** Sequence.
*/
typedef std::vector<Event> Sequence;
typedef std::vector<StringEvent> StringSequence;

/** TriSequence.
	@class TriSequence
*/
class TriSequence {
public:
	Event topic;
	Sequence seq;
	size_t size() { return seq.size(); };
};

class TriStringSequence {
public:
	Event topic;
	StringSequence seq;
	size_t size() { return seq.size(); };
};

/**	Data.
	A vector that contains a collection of event.
	@class Data
*/
template <typename T = Sequence>
class Data : public std::vector<T> {
private:
	size_t n_element;
public:
	void append(T element) { this->push_back(element); n_element += element.size(); };
	size_t size_element() { return n_element; };
};

} // namespace tricrf

#endif

