/*****************************************************************************
FILE:  analbeat.cpp
AUTHOR:	Patrick S. Hamilton
REVISED:	5/13/2002
  ___________________________________________________________________________
analbeat.cpp: Analyze Beat
Copywrite (C) 2001 Patrick S. Hamilton
This file is free software; you can redistribute it and/or modify it under
the terms of the GNU Library General Public License as published by the Free
Software Foundation; either version 2 of the License, or (at your option) any
later version.

This software is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Library General Public License for more
details.

You should have received a copy of the GNU Library General Public License along
with this library; if not, write to the Free Software Foundation, Inc., 59
Temple Place - Suite 330, Boston, MA 02111-1307, USA.

You may contact the author by e-mail (pat@eplimited.edu) or postal mail
(Patrick Hamilton, E.P. Limited, 35 Medford St., Suite 204 Somerville,
MA 02143 USA).  For updates to this software, please visit our website
(http://www.eplimited.com).
  __________________________________________________________________________

  This file contains functions for determining the QRS onset, QRS offset,
  beat onset, beat offset, polarity, and isoelectric level for a beat.

  Revisions:
	4/16: Modified to prevent isoStart from being set to less than ISO_LENGTH1-1
   5/13/2002: Time related constants are tied to BEAT_SAMPLE_RATE in bdac.h.

*****************************************************************************/
#include "bdac.h"
#include <stdio.h>
#include <stdlib.h>

#define ISO_LENGTH1  BEAT_MS50
#define ISO_LENGTH2	BEAT_MS80
#define ISO_LIMIT	20

#include "tsc_x86.h"

#ifdef OPERATION_COUNTER
extern long int float_add_counter;
extern long int float_mul_counter;
extern long int float_div_counter;
extern long int float_comp_counter;
#endif

// Local prototypes.

inline int IsoCheck(float *data, int isoLength);

/****************************************************************
	IsoCheck determines whether the amplitudes of a run
	of data fall within a sufficiently small amplitude that
	the run can be considered isoelectric.
*****************************************************************/

inline int IsoCheck(float *data, int isoLength)
{
	int i ;
	float max, min;
	for(i = 1, max=min = data[0]; i < isoLength; ++i)
	{
		#ifdef OPERATION_COUNTER 
		float_comp_counter+=2;
		#endif
		if(data[i] > max){
			max = data[i] ;
			#ifdef OPERATION_COUNTER 
			float_comp_counter--;
			#endif
		} else if(data[i] < min){
			min = data[i] ;
		}
	}

	#ifdef OPERATION_COUNTER 
		float_add_counter++;
		float_comp_counter++;
	#endif
		
	if(max - min < (float)ISO_LIMIT)
		return(1) ;
  	return(0) ;
}

/**********************************************************************
	AnalyzeBeat takes a beat buffer as input and returns (via pointers)
	estimates of the QRS onset, QRS offset, polarity, isoelectric level
	beat beginning (P-wave onset), and beat ending (T-wave offset).
	Analyze Beat assumes that the beat has been sampled at 100 Hz, is
	BEATLGTH long, and has an R-wave location of roughly FIDMARK.

	Note that beatBegin is the number of samples before FIDMARK that
	the beat begins and beatEnd is the number of samples after the
	FIDMARK that the beat ends.
************************************************************************/

#define INF_CHK_N	BEAT_MS40

void AnalyzeBeat(float *beat, int *onset, int *offset, int *isoLevel,
	int *beatBegin, int *beatEnd, int *amp)
	{
	float maxSlope = 0.0f, minSlope = 0.0f;
	int maxSlopeI, minSlopeI;
	float maxV, minV ;
	int isoStart, isoEnd ;
	float slope;
	int i ;

	// Search back from the fiducial mark to find the isoelectric
	// region preceeding the QRS complex.

	for(i = FIDMARK-ISO_LENGTH2; (i > 0) && (IsoCheck(&beat[i],ISO_LENGTH2) == 0); --i) ;

	// If the first search didn't turn up any isoelectric region, look for
	// a shorter isoelectric region.

	if(i == 0)
		{
		for(i = FIDMARK-ISO_LENGTH1; (i > 0) && (IsoCheck(&beat[i],ISO_LENGTH1) == 0); --i) ;
		isoStart = i + (ISO_LENGTH1 - 1) ;
		}
	else isoStart = i + (ISO_LENGTH2 - 1) ;

	// Search forward from the R-wave to find an isoelectric region following
	// the QRS complex.

	for(i = FIDMARK; (i < BEATLGTH) && (IsoCheck(&beat[i],ISO_LENGTH1) == 0); ++i) ;
	isoEnd = i ;

	// Find the maximum and minimum slopes on the
	// QRS complex.

	i = FIDMARK-BEAT_MS150 ;
	maxSlope = maxSlope = beat[i] - beat[i-1] ;
	#ifdef OPERATION_COUNTER 
		float_add_counter++;
	#endif
	maxSlopeI = minSlopeI = i ;

	for(; i < FIDMARK+BEAT_MS150; ++i)
		{
		slope = beat[i] - beat[i-1] ;
		#ifdef OPERATION_COUNTER 
			float_add_counter++;
			float_comp_counter++;
		#endif
		if(slope > maxSlope)
			{
			maxSlope = slope ;
			maxSlopeI = i ;
			}
		else if(slope < minSlope)
			{
			minSlope = slope ;
			minSlopeI = i ;
			#ifdef OPERATION_COUNTER 
			float_comp_counter++;
			#endif
			}
		#ifdef OPERATION_COUNTER 
		else
			{
			float_comp_counter++;
			}
		#endif
		}

	// Use the smallest of max or min slope for search parameters.

	if(maxSlope > -minSlope)
		maxSlope = -minSlope ;
	else minSlope = -maxSlope ;
	#ifdef OPERATION_COUNTER 
		float_add_counter += 2; // because it is once for the if and once in the if
		float_comp_counter+= 2; // one for the one above, one for bellow
	#endif

	if(maxSlopeI < minSlopeI)
		{

		// Search back from the maximum slope point for the QRS onset.

	for(i = maxSlopeI;(i > 0) && ((beat[i]-beat[i-1]) > (maxSlope/4.0f)); --i){
		#ifdef OPERATION_COUNTER
			float_div_counter++;
			float_comp_counter++;
		#endif
		}
	#ifdef OPERATION_COUNTER
		float_div_counter++;
		float_comp_counter++;
	#endif
	*onset = i-1 ;

		// Check to see if this was just a brief inflection.

		for(; (i > *onset-INF_CHK_N) && ((beat[i]-beat[i-1]) <= (maxSlope/4.0f)); --i){
			#ifdef OPERATION_COUNTER
				float_div_counter++;
				float_add_counter++;
				float_comp_counter++;
			#endif
		}
		#ifdef OPERATION_COUNTER
			float_div_counter++;
			float_add_counter++;
			float_comp_counter++;
		#endif

		if(i > *onset-INF_CHK_N)
			{
			for(;(i > 0) && ((beat[i]-beat[i-1]) > (maxSlope/4.0f)); --i){
				#ifdef OPERATION_COUNTER
					float_div_counter++;
					float_add_counter++;
					float_comp_counter++;
				#endif
			}
			#ifdef OPERATION_COUNTER
				float_div_counter++;
				float_add_counter++;
				float_comp_counter++;
			#endif
			*onset = i-1 ;
			}
		i = *onset+1 ;

		// Check to see if a large negative slope follows an inflection.
		// If so, extend the onset a little more.

		for(;(i > *onset-INF_CHK_N) && ((beat[i-1]-beat[i]) < (maxSlope/4.0f)); --i){
			#ifdef OPERATION_COUNTER
				float_div_counter++;
				float_add_counter++;
				float_comp_counter++;
			#endif
		}
		#ifdef OPERATION_COUNTER
			float_div_counter++;
			float_add_counter++;
			float_comp_counter++;
		#endif

		if(i > *onset-INF_CHK_N)
			{
			for(; (i > 0) && ((beat[i-1]-beat[i]) > (maxSlope/4.0f)); --i){
				#ifdef OPERATION_COUNTER
					float_div_counter++;
					float_add_counter++;
					float_comp_counter++;
				#endif
			}
			#ifdef OPERATION_COUNTER
				float_div_counter++;
				float_add_counter++;
				float_comp_counter++;
			#endif
			*onset = i-1 ;
			}

		// Search forward from minimum slope point for QRS offset.

		for(i = minSlopeI; (i < BEATLGTH) && ((beat[i] - beat[i-1]) < (minSlope /4.0f)); ++i){
			#ifdef OPERATION_COUNTER
				float_div_counter++;
				float_add_counter++;
				float_comp_counter++;
			#endif
		}
		#ifdef OPERATION_COUNTER
			float_div_counter++;
			float_add_counter++;
			float_comp_counter++;
		#endif
			*offset = i ;

		// Make sure this wasn't just an inflection.

		for(; (i < *offset+INF_CHK_N) && ((beat[i]-beat[i-1]) >= (minSlope/4.0f)); ++i){
			#ifdef OPERATION_COUNTER
				float_div_counter++;
				float_add_counter++;
				float_comp_counter++;
			#endif
		}
		#ifdef OPERATION_COUNTER
			float_div_counter++;
			float_add_counter++;
			float_comp_counter++;
		#endif

		if(i < *offset+INF_CHK_N)
			{
			for(;(i < BEATLGTH) && ((beat[i]-beat[i-1]) < (minSlope /4.0f)); ++i){
				#ifdef OPERATION_COUNTER
					float_div_counter++;
					float_add_counter++;
					float_comp_counter++;
				#endif
			}
			#ifdef OPERATION_COUNTER
				float_div_counter++;
				float_add_counter++;
				float_comp_counter++;
			#endif
			*offset = i ;
			}
		i = *offset ;

		// Check to see if there is a significant upslope following
		// the end of the down slope.

		for(;(i < *offset+BEAT_MS40) && ((beat[i-1]-beat[i]) > (minSlope/4.0f)); ++i){
			#ifdef OPERATION_COUNTER
				float_div_counter++;
				float_add_counter++;
				float_comp_counter++;
			#endif
		}
		#ifdef OPERATION_COUNTER
			float_div_counter++;
			float_add_counter++;
			float_comp_counter++;
		#endif
		if(i < *offset+BEAT_MS40)
			{
			for(; (i < BEATLGTH) && ((beat[i-1]-beat[i]) < (minSlope/4.0f)); ++i){
				#ifdef OPERATION_COUNTER
					float_div_counter++;
					float_add_counter++;
					float_comp_counter++;
				#endif
			}
			#ifdef OPERATION_COUNTER
				float_div_counter++;
				float_add_counter++;
				float_comp_counter++;
			#endif
			*offset = i ;

			// One more search motivated by PVC shape in 123.

			for(; (i < *offset+BEAT_MS60) && (beat[i]-beat[i-1] > (minSlope/4.0f)); ++i){
				#ifdef OPERATION_COUNTER
					float_div_counter++;
					float_add_counter++;
					float_comp_counter++;
				#endif
			}
			#ifdef OPERATION_COUNTER
				float_div_counter++;
				float_add_counter++;
				float_comp_counter++;
			#endif
			if(i < *offset + BEAT_MS60)
				{
				for(;(i < BEATLGTH) && (beat[i]-beat[i-1] < (minSlope/4.0f)); ++i){
					#ifdef OPERATION_COUNTER
						float_div_counter++;
						float_add_counter++;
						float_comp_counter++;
					#endif
				}
				#ifdef OPERATION_COUNTER
					float_div_counter++;
					float_add_counter++;
					float_comp_counter++;
				#endif
				*offset = i ;
				}
			}
		}
	else
		{

		// Search back from the minimum slope point for the QRS onset.

		for(i = minSlopeI;(i > 0) && ((beat[i]-beat[i-1]) < (minSlope/4.0f)); --i){
				#ifdef OPERATION_COUNTER
					float_div_counter++;
					float_add_counter++;
					float_comp_counter++;
				#endif
			}
			#ifdef OPERATION_COUNTER
				float_div_counter++;
				float_add_counter++;
				float_comp_counter++;
			#endif
			*onset = i-1 ;

		// Check to see if this was just a brief inflection.

		for(; (i > *onset-INF_CHK_N) && ((beat[i]-beat[i-1]) >= (minSlope/4.0f)); --i){
			#ifdef OPERATION_COUNTER
				float_div_counter++;
				float_add_counter++;
				float_comp_counter++;
			#endif
		}
		#ifdef OPERATION_COUNTER
			float_div_counter++;
			float_add_counter++;
			float_comp_counter++;
		#endif
		if(i > *onset-INF_CHK_N)
			{
			for(; (i > 0) && ((beat[i]-beat[i-1]) < (minSlope/4.0f));--i){
				#ifdef OPERATION_COUNTER
					float_div_counter++;
					float_add_counter++;
					float_comp_counter++;
				#endif
			}
			#ifdef OPERATION_COUNTER
				float_div_counter++;
				float_add_counter++;
				float_comp_counter++;
			#endif
			*onset = i-1 ;
			}
		i = *onset+1 ;

		// Check for significant positive slope after a turning point.

		for(;(i > *onset-INF_CHK_N) && ((beat[i-1]-beat[i]) > (minSlope/4.0f)); --i){
			#ifdef OPERATION_COUNTER
				float_div_counter++;
				float_add_counter++;
				float_comp_counter++;
			#endif
		}
		#ifdef OPERATION_COUNTER
			float_div_counter++;
			float_add_counter++;
			float_comp_counter++;
		#endif
		if(i > *onset-INF_CHK_N)
			{
			for(; (i > 0) && ((beat[i-1]-beat[i]) < (minSlope/4.0f)); --i){
				#ifdef OPERATION_COUNTER
					float_div_counter++;
					float_add_counter++;
					float_comp_counter++;
				#endif	
			}
			#ifdef OPERATION_COUNTER
				float_div_counter++;
				float_add_counter++;
				float_comp_counter++;
			#endif
			*onset = i-1 ;
			}

		// Search forward from maximum slope point for QRS offset.

		for(i = maxSlopeI;(i < BEATLGTH) && ((beat[i] - beat[i-1]) > (maxSlope/4.0f)); ++i){
			#ifdef OPERATION_COUNTER
				float_div_counter++;
				float_add_counter++;
				float_comp_counter++;
			#endif
		}
		#ifdef OPERATION_COUNTER
			float_div_counter++;
			float_add_counter++;
			float_comp_counter++;
		#endif
		*offset = i ;

		// Check to see if this was just a brief inflection.

		for(; (i < *offset+INF_CHK_N) && ((beat[i] - beat[i-1]) <= (maxSlope/4.0f)); ++i){
			#ifdef OPERATION_COUNTER
				float_div_counter++;
				float_add_counter++;
				float_comp_counter++;
			#endif
		}
		#ifdef OPERATION_COUNTER
			float_div_counter++;
			float_add_counter++;
			float_comp_counter++;
		#endif
		if(i < *offset+INF_CHK_N)
			{
			for(;(i < BEATLGTH) && ((beat[i] - beat[i-1]) > (maxSlope/4.0f)); ++i){
				#ifdef OPERATION_COUNTER
					float_div_counter++;
					float_add_counter++;
					float_comp_counter++;
				#endif
			}
			#ifdef OPERATION_COUNTER
				float_div_counter++;
				float_add_counter++;
				float_comp_counter++;
			#endif
			*offset = i ;
			}
		i = *offset ;

		// Check to see if there is a significant downslope following
		// the end of the up slope.

		for(;(i < *offset+BEAT_MS40) && ((beat[i-1]-beat[i]) < (maxSlope/4.0f)); ++i){
			#ifdef OPERATION_COUNTER
				float_div_counter++;
				float_add_counter++;
				float_comp_counter++;
			#endif
		}
		#ifdef OPERATION_COUNTER
			float_div_counter++;
			float_add_counter++;
			float_comp_counter++;
		#endif
		if(i < *offset+BEAT_MS40)
			{
			for(; (i < BEATLGTH) && ((beat[i-1]-beat[i]) > (maxSlope/4.0f)); ++i){
				#ifdef OPERATION_COUNTER
					float_div_counter++;
					float_add_counter++;
					float_comp_counter++;
				#endif
			}
			#ifdef OPERATION_COUNTER
				float_div_counter++;
				float_add_counter++;
				float_comp_counter++;
			#endif
			*offset = i ;
			}
		}

	// If the estimate of the beginning of the isoelectric level was
	// at the beginning of the beat, use the slope based QRS onset point
	// as the iso electric point.

	if((isoStart == ISO_LENGTH1-1)&& (*onset > isoStart)) // ** 4/19 **
		isoStart = *onset ;

	// Otherwise, if the isoelectric start and the slope based points
	// are close, use the isoelectric start point.

	else if(*onset-isoStart < BEAT_MS50)
		*onset = isoStart ;

	// If the isoelectric end and the slope based QRS offset are close
	// use the isoelectic based point.

	if(isoEnd - *offset < BEAT_MS50)
		*offset = isoEnd ;

	*isoLevel = beat[isoStart] ;


	// Find the maximum and minimum values in the QRS.

	for(i = *onset, maxV = minV = beat[*onset]; i < *offset; ++i){
		#ifdef OPERATION_COUNTER
			float_comp_counter++;
		#endif
		if(beat[i] > maxV){
			maxV = beat[i] ;
		} else if(beat[i] < minV){
			minV = beat[i] ;
			#ifdef OPERATION_COUNTER
				float_comp_counter++;
			#endif
		}
		#ifdef OPERATION_COUNTER
		else{
			float_comp_counter++;
		}
		#endif
	}
	#ifdef OPERATION_COUNTER
		float_comp_counter++;
	#endif

	// If the offset is significantly below the onset and the offset is
	// on a negative slope, add the next up slope to the width.

	#ifdef OPERATION_COUNTER
		float_div_counter+=2;
		float_add_counter++;
	#endif

	if((beat[*onset]-beat[*offset] > ((maxV-minV)/4.0f)+((maxV-minV)/8.0f)))
		{

		// Find the maximum slope between the finish and the end of the buffer.

		for(i = maxSlopeI = *offset, maxSlope = beat[*offset] - beat[*offset-1];
			(i < *offset+BEAT_MS100) && (i < BEATLGTH); ++i)
			{
			slope = beat[i]-beat[i-1] ;
			if(slope > maxSlope)
				{
				maxSlope = slope ;
				maxSlopeI = i ;
				}
			#ifdef OPERATION_COUNTER
				float_add_counter+=2;
				float_comp_counter++;
			#endif
			}

		// Find the new offset.

		if(maxSlope > 0){
			for(i = maxSlopeI;(i < BEATLGTH) && (beat[i]-beat[i-1] > (maxSlope/2.0f)); ++i){
				#ifdef OPERATION_COUNTER
					float_div_counter++;
					float_add_counter++;
					float_comp_counter++;
				#endif
				}
			*offset = i ;
			}
			#ifdef OPERATION_COUNTER
				float_div_counter++;
				float_add_counter++;
				float_comp_counter++;
			#endif
		}

	// Determine beginning and ending of the beat.
	// Search for an isoelectric region that precedes the R-wave.
	// by at least 250 ms.

	for(i = FIDMARK-BEAT_MS250;
		(i >= BEAT_MS80) && (IsoCheck(&beat[i-BEAT_MS80],BEAT_MS80) == 0); --i) ;
	*beatBegin = i ;

	// If there was an isoelectric section at 250 ms before the
	// R-wave, search forward for the isoelectric region closest
	// to the R-wave.  But leave at least 50 ms between beat begin
	// and onset, or else normal beat onset is within PVC QRS complexes.
	// that screws up noise estimation.

	if(*beatBegin == FIDMARK-BEAT_MS250)
		{
		for(; (i < *onset-BEAT_MS50) &&
			(IsoCheck(&beat[i-BEAT_MS80],BEAT_MS80) == 1); ++i) ;
		*beatBegin = i-1 ;
		}

	// Rev 1.1
	else if(*beatBegin == BEAT_MS80 - 1)
		{
		for(; (i < *onset) && (IsoCheck(&beat[i-BEAT_MS80],BEAT_MS80) == 0); ++i);
		if(i < *onset)
			{
			for(; (i < *onset) && (IsoCheck(&beat[i-BEAT_MS80],BEAT_MS80) == 1); ++i) ;
			if(i < *onset)
				*beatBegin = i-1 ;
			}
		}

	// Search for the end of the beat as the first isoelectric
	// segment that follows the beat by at least 300 ms.

	for(i = FIDMARK+BEAT_MS300;
		(i < BEATLGTH) && (IsoCheck(&beat[i],BEAT_MS80) == 0); ++i) ;
	*beatEnd = i ;

	// If the signal was isoelectric at 300 ms. search backwards.
/*	if(*beatEnd == FIDMARK+30)
		{
		for(; (i > *offset) && (IsoCheck(&beat[i],8) != 0); --i) ;
		*beatEnd = i ;
		}
*/
	// Calculate beat amplitude.

	maxV=minV=beat[*onset] ;
	for(i = *onset; i < *offset; ++i){
		#ifdef OPERATION_COUNTER
			float_comp_counter++;
		#endif
		if(beat[i] > maxV){
			maxV = beat[i] ;
		}
		else if(beat[i] < minV){
			minV = beat[i] ;
			#ifdef OPERATION_COUNTER
				float_comp_counter++;
			#endif
		}
		#ifdef OPERATION_COUNTER
		else {
			float_comp_counter++;
		}
		#endif
	}
	*amp = maxV-minV ;

	#ifdef OPERATION_COUNTER
		float_add_counter++;
	#endif

	}
