#pragma once

#include "Solver.h"
#include "catpaw/cpToolBox.h"
#include "particle_common.h"

class ParticleObject {
public:
	vecf3 pos;
	vecf3 normal;
	vecf volfrac;
	veci type;
};

class FluidGenerator {

};

class BoundaryGenerator {
private:
	ParticleObject* so;
	Tinyxml_Reader reader;
public:
	void parseNode(XMLElement* e) {
		reader.Use(e);
		if (strcmp(e->Name(), "Cube")==0) {
			cfloat3 xmin = reader.GetFloat3("xmin");
			cfloat3 xmax = reader.GetFloat3("xmax");
			cfloat3 normal = reader.GetFloat3("normal");
			float spacing = reader.GetFloat("spacing");

			for (float x=xmin.x; x<=xmax.x; x+=spacing)
				for (float y=xmin.y; y<=xmax.y; y+=spacing)
					for (float z=xmin.z; z<=xmax.z; z+=spacing) {
						so->pos.push_back(cfloat3(x, y, z));
						so->normal.push_back(normal);
						so->type.push_back(TYPE_BOUNDARY);
					}
		}
	}

	ParticleObject* loadxml(const char* filepath) {
		so = new ParticleObject();

		printf("Parsing XML.\n");
		tinyxml2::XMLDocument doc;
		int result = doc.LoadFile(filepath);
		Tinyxml_Reader reader;
		XMLElement* boundElement = doc.FirstChildElement("Boundary");
		XMLElement* child = boundElement->FirstChildElement();
		while (child) {
			parseNode(child);
			child = child->NextSiblingElement();
		}
		return so;
	}
};
