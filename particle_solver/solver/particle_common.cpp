
#include "particle_common.h"
#include "sph_solver.cuh"

extern SimParam_SPH hParam;

void loadFluidVolume(XMLElement* sceneEle, int typenum, vector<fluidvol>& fvs) {

	if (sceneEle == NULL) {
		printf("error loading fluid volume: invalid scene element.\n");
		return;
	}

	Tinyxml_Reader reader;
	fluidvol fv;
	XMLElement* fvele = sceneEle->FirstChildElement("Vol");
	const char* tmp;
	const char* empty;

	while (fvele!=NULL && strcmp(fvele->Name(), "Vol")==0) {
		reader.Use(fvele);
		fv.xmin = reader.GetFloat3("VolMin");
		fv.xmax = reader.GetFloat3("VolMax");
		fv.group = reader.GetInt("Group");
		tmp = reader.GetText("Type");
		empty = reader.GetText("Empty");
		int layer = reader.GetInt("Layer");
		
		//printf("%s %d\n", tmp, strcmp(tmp,"deformable"));

		if(!tmp)
			fv.type = TYPE_FLUID;
		else if(strcmp(tmp,"fluid")==0)
			fv.type = TYPE_FLUID;
		else if(strcmp(tmp,"deformable")==0)
			fv.type = TYPE_DEFORMABLE;
		else if(strcmp(tmp,"granular")==0)
			fv.type = TYPE_GRANULAR;

		if (!empty || strcmp(empty, "false") == 0)
			fv.empty = false;
		else
			fv.empty = true;

		reader.GetFloatN(fv.volfrac, typenum, "VolFrac");
		fvs.push_back(fv);

		if (fv.type == TYPE_DEFORMABLE && layer > 1)
		{
			float spacing = hParam.spacing;
			for (int i = 1; i < layer; i++)
			{
				fluidvol morefv;
				morefv.xmin = fv.xmin + cfloat3(spacing, spacing, spacing) * i;
				morefv.xmax = fv.xmax - cfloat3(spacing, spacing, spacing) * i;
				morefv.group = fv.group;
				morefv.type = TYPE_DEFORMABLE;
				morefv.empty = true;
				for (int i = 0; i < 10; i++)
					morefv.volfrac[i] = fv.volfrac[i];
				fvs.push_back(morefv);
			}
		}

		fvele = fvele->NextSiblingElement();
	}
}