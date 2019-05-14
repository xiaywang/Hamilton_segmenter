void ResetPostClassify() ;
void PostClassify(int *recentTypes, int domType, int *recentRRs, int width, double mi2,
	int rhythmClass) ;
int CheckPostClass(int type) ;
int CheckPCRhythm(int type) ;

int PostClass[MAXTYPES][8];
int PCRhythm[MAXTYPES][8];
int PCInitCount;
