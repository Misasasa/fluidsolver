#pragma once
#include "tinyxml2.h"
#include <string>


using namespace tinyxml2;

cfloat3 QueryFloat3(XMLElement* element);
float XMLGetFloat(const char* name);
int XMLGetInt(const char* name);
cfloat3& XMLGetFloat3(const char* name);
void XMLGetFloatN(float* buffer, int size, const char* name);

struct Tinyxml_Reader {
	XMLElement* basenode = NULL;

	void Use(XMLElement* base);
	const char* GetText(const char* name);
	float GetFloat(const char* name);
	int GetInt(const char* name);
	cfloat3& GetFloat3(const char* name);
	void GetFloatN(float* buffer, int size, const char* name);
};