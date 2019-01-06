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
	veci id;
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
			int id = reader.GetInt("id");


			for (float x=xmin.x; x<=xmax.x; x+=spacing)
				for (float y=xmin.y; y<=xmax.y; y+=spacing)
					for (float z=xmin.z; z<=xmax.z; z+=spacing) {
						so->pos.push_back(cfloat3(x, y, z));
						so->normal.push_back(normal);
						so->type.push_back(TYPE_RIGID);
						so->id.push_back(id);
					}
		}
		else if (strcmp(e->Name(), "Box")==0) {
			cfloat3 xmin = reader.GetFloat3("xmin");
			cfloat3 xmax = reader.GetFloat3("xmax");
			cfloat3 normal = reader.GetFloat3("normal");
			float spacing = reader.GetFloat("spacing");
			float thickness = reader.GetFloat("thickness");
			int id = reader.GetInt("id");

			//x-y
			for (float x=xmin.x; x<=xmax.x; x+=spacing)
				for (float y=xmin.y; y<=xmax.y; y+=spacing){
						so->pos.push_back(cfloat3(x, y, xmin.z));
						so->normal.push_back(cfloat3(0,0,1));
						so->type.push_back(TYPE_RIGID);
						so->id.push_back(id);

						so->pos.push_back(cfloat3(x, y, xmax.z));
						so->normal.push_back(cfloat3(0, 0, -1));
						so->type.push_back(TYPE_RIGID);
						so->id.push_back(id);
				}
			//y-z
			for (float z=xmin.z; z<=xmax.z; z+=spacing)
				for (float y=xmin.y; y<=xmax.y; y+=spacing) {
					so->pos.push_back(cfloat3(xmin.x, y, z));
					so->normal.push_back(cfloat3(1, 0, 0));
					so->type.push_back(TYPE_RIGID);
					so->id.push_back(id);

					so->pos.push_back(cfloat3(xmax.x, y, z));
					so->normal.push_back(cfloat3(-1, 0, 0));
					so->type.push_back(TYPE_RIGID);
					so->id.push_back(id);
				}
			//x-z
			for (float x=xmin.x; x<=xmax.x; x+=spacing)
				for (float z=xmin.z; z<=xmax.z; z+=spacing) {
					so->pos.push_back(cfloat3(x, xmin.y, z));
					so->normal.push_back(cfloat3(0, 1, 0));
					so->type.push_back(TYPE_RIGID);
					so->id.push_back(id);

					so->pos.push_back(cfloat3(x, xmax.y, z));
					so->normal.push_back(cfloat3(0, -1, 0));
					so->type.push_back(TYPE_RIGID);
					so->id.push_back(id);
				}
					
		}
		else if (strcmp(e->Name(), "OpenBox")==0) {
			cfloat3 xmin = reader.GetFloat3("xmin");
			cfloat3 xmax = reader.GetFloat3("xmax");
			cfloat3 normal = reader.GetFloat3("normal");
			float spacing = reader.GetFloat("spacing");
			float thickness = reader.GetFloat("thickness");
			int id = reader.GetInt("id");

			//x-y
			for (float x=xmin.x; x<=xmax.x; x+=spacing)
				for (float y=xmin.y; y<=xmax.y; y+=spacing) {
					so->pos.push_back(cfloat3(x, y, xmin.z));
					so->normal.push_back(cfloat3(0, 0, 1));
					so->type.push_back(TYPE_RIGID);
					so->id.push_back(id);

					so->pos.push_back(cfloat3(x, y, xmax.z));
					so->normal.push_back(cfloat3(0, 0, -1));
					so->type.push_back(TYPE_RIGID);
					so->id.push_back(id);
				}
			//y-z
			for (float z=xmin.z; z<=xmax.z; z+=spacing)
				for (float y=xmin.y; y<=xmax.y; y+=spacing) {
					so->pos.push_back(cfloat3(xmin.x, y, z));
					so->normal.push_back(cfloat3(1, 0, 0));
					so->type.push_back(TYPE_RIGID);
					so->id.push_back(id);

					so->pos.push_back(cfloat3(xmax.x, y, z));
					so->normal.push_back(cfloat3(-1, 0, 0));
					so->type.push_back(TYPE_RIGID);
					so->id.push_back(id);
				}
			//x-z
			for (float x=xmin.x; x<=xmax.x; x+=spacing)
				for (float z=xmin.z; z<=xmax.z; z+=spacing) {
					so->pos.push_back(cfloat3(x, xmin.y, z));
					so->normal.push_back(cfloat3(0, 1, 0));
					so->type.push_back(TYPE_RIGID);
					so->id.push_back(id);

					/*so->pos.push_back(cfloat3(x, xmax.y, z));
					so->normal.push_back(cfloat3(0, -1, 0));
					so->type.push_back(TYPE_RIGID);
					so->id.push_back(id);*/
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
