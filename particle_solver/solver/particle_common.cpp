
#include "particle_common.h"

void loadFluidVolume(XMLElement* sceneEle, int typenum, vector<fluidvol>& fvs) {

	Tinyxml_Reader reader;
	fluidvol fv;
	XMLElement* fvele = sceneEle->FirstChildElement("Vol");
	while (fvele!=NULL && strcmp(fvele->Name(), "Vol")==0) {
		reader.Use(fvele);
		fv.xmin = reader.GetFloat3("VolMin");
		fv.xmax = reader.GetFloat3("VolMax");
		reader.GetFloatN(fv.volfrac, typenum, "VolFrac");
		fvs.push_back(fv);
		fvele = fvele->NextSiblingElement();
	}
}