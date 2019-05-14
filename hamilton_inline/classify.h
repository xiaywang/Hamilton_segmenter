#define DM_BUFFER_LENGTH	180
extern int DMBeatTypes[DM_BUFFER_LENGTH];
extern int DMNormCounts[8], DMBeatCounts[8];
inline int HFNoiseCheck(float *beat) ;
int runCount;
int DMBeatTypes[DM_BUFFER_LENGTH], DMBeatClasses[DM_BUFFER_LENGTH] ;
int DMNormCounts[8], DMBeatCounts[8], DMIrregCount ;
