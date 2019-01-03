
#include "particle_common.h"

void loadFluidVolume(XMLElement* sceneEle, int typenum, vector<fluidvol>& fvs) {

	if (sceneEle == NULL) {
		printf("error loading fluid volume: invalid scene element.\n");
		return;
	}

	Tinyxml_Reader reader;
	fluidvol fv;
	XMLElement* fvele = sceneEle->FirstChildElement("Vol");
	const char* tmp;

	while (fvele!=NULL && strcmp(fvele->Name(), "Vol")==0) {
		reader.Use(fvele);
		fv.xmin = reader.GetFloat3("VolMin");
		fv.xmax = reader.GetFloat3("VolMax");
		fv.group = reader.GetInt("Group");
		tmp = reader.GetText("Type");

		if(!tmp)
			fv.type = TYPE_FLUID;
		else if(strcmp(tmp,"fluid")==0)
			fv.type = TYPE_FLUID;
		else if(strcmp(tmp,"deformable")==0)
			fv.type = TYPE_DEFORMABLE;
		else if(strcmp(tmp,"granular")==0)
			fv.type = TYPE_GRANULAR;

		reader.GetFloatN(fv.volfrac, typenum, "VolFrac");
		fvs.push_back(fv);
		fvele = fvele->NextSiblingElement();
	}
}