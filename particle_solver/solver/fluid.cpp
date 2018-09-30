
#include "cuda.h"
#include "cuda_runtime.h"
#include "host_defines.h"

#include "pbfsolver.h"




extern SimParam hParam;



void PBFSolver::loadFluidVols(XMLElement* sceneEle) {
	Tinyxml_Reader reader;
	fluidvol fv;
	XMLElement* fvele = sceneEle->FirstChildElement("Vol");
	while (fvele!=NULL && strcmp(fvele->Name(), "Vol")==0) {
		reader.Use(fvele);
		fv.xmin = reader.GetFloat3("VolMin");
		fv.xmax = reader.GetFloat3("VolMax");
		reader.GetFloatN(fv.volfrac, hParam.maxtypenum, "VolFrac");
		fluidvols.push_back(fv);
		fvele = fvele->NextSiblingElement();
	}
}
