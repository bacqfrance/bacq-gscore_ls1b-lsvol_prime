/**  ----------------------------
@file simpleSimulatedAnnealing.cpp
@brief Optimization heuristics implemented in C++ especially for CPU/QPU comparison
Provides the implementation of function sets for implementing CPU/QPU 
comparison benchmarks for optimization heuristics. It provides both a usual binary
background and a Ising implementation as well.<br/>
Dependencies are:
	- stdc++ (C++20)
	- openmp (recomended)
	@copyright (c) CEA LIST, QuLIST team <br/>
	Distributed under EU Public Licence (EUPL) <br/>
@verbatim
See Marc C. Robini, Thierry Rastello, Isabelle E. Magnin
“Simulated Annealing, Acceleration Techniques, and Image Restoration”
IEEE Transactions on Image Processing, Vol. 8, No. 10, 1999, pp. 1374–1387.
For justifications of the implementation (geometric cooling)
@endverbatim
--------------------------------*/

#include <iostream>
#include <assert.h>
#include <chrono>
#include <random>
#include <math.h>
#include <cmath>
#include <vector>
#include <array>
#include <limits>
#include <cstdlib> // Required for rand() and srand()
#include <ctime>   // Required for time()
#ifdef USE_OPENMP
    #include <omp.h>
    #define OMP_PARALLEL _Pragma("omp parallel")
#else
    #define OMP_PARALLEL
#endif

// Random number generators: standard functions

// uniform: return a double in the interval (0,1)
double uniform(void)
{
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 1.0);
    return dis(gen);
}


// uniformDist: returns an integer in the range 0 to maxRand (the latest not included)
int uniformDist(int maxRand) {
    // 1. Obtain a seed for the random number engine
    // E.G. Using a time-based seed for a different sequence each run
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    // 2. Standard mersenne_twister engine which is a high-quality pseudo-random generator
    //   this modus operandi can be considered as standard for random number generation
    std::mt19937 mt(seed); 

    // 3. Define the desired distribution (e.g., uniform integer distribution from 0 to 99 included)
    std::uniform_int_distribution<int> dist(0, maxRand-1); 

    // 4. Generate the random number
    int random_num = dist(mt);
    
    // return random number between 0 and maxRand-1
    return random_num;
}

// types of problems to solve
enum ModelType { BINARY, SPIN };


// definition of Ising Model class
class IsingModel {
	private:
		ModelType isingType= SPIN;
		int defaultModelVal= 0;
		int isingSize= 0;
		std::vector<int> *currentConfig= nullptr; // spin configuration
		std::vector<int> *previousConfig= nullptr; // previous sping configuration
		std::vector<double> *myh= nullptr;
		std::vector<double> *myJ= nullptr;
		double currentEnergy= std::numeric_limits<double>::infinity();
		double previousEnergy= std::numeric_limits<double>::infinity();
		void initRandomConfig(std::vector<int>*configuration);
		double getEnergy(std::vector<int> *isingConfiguration);
	public:
		IsingModel(int size, ModelType mtype, std::vector<int>*buf1, std::vector<int>*buf2);
		void initModel(std::vector<double>*J, std::vector<double>*h); // coupling matrix isingSize*isingSize J and magnetic vector h 
		int flip(int val); // flip value of spin according to model
		double getCurrentEnergy(void);
		double getPreviousEnergy(void);
		void generateCandidateConfiguration(void);
		void selectCurrent(void); // use current trial configuration as reference
		void discardCurrent(void); // revert to previous trial configuration
		std::vector<int>*getCurrentConfig(void);
} ;


// Definition for class SimulatedAnnealing. Class IsingModel is a prerequisite as some data are of this type
class SimulatedAnnealing {
	private:
		IsingModel *myIsing= nullptr; // associated Ising model
		int Isize= 0;                 // and its size
		double initTemperature= 100.0; // starting temperature of the heuristic (default value)
		double endingTemperature= 1e-6 ; // ending temperature of the heuristic
		double deltaTemperature= 0.95;  // scale factor for decreasing temperature (<1)
		double scale= 10.0;            // scale for temperature accounting (random selection)
		int numSwipe= 200;             // number of swipes per temperature level
		std::array<std::vector<int>,3> configbuf;  // configuration buffers for the models (size Isize)
		double bestEnergy= std::numeric_limits<double>::infinity();  // best energy found in the heuristic
	public:
		SimulatedAnnealing(int size, ModelType mtype);  // constructor size of the samples and the h vector. type of model 
		void initCoupling(std::vector<double>*J, std::vector<double>*h); // set coupling values matrix J and vector h
		void run(void);                     // run the heuristic
		void setInitialTemperature(double temp);  // set new value of initial temperature
		std::vector<int> *getCurrentConfig(void);  // obtain current configuration
		void setScale(double scale);              // set new scale value for the heuristic
		void setSwipeNumber(int nswipe);	// set new number of swipes per temperature level
		void setCurrentAsBest(void);        // select current configuration of the Ising model as the best candidate
		std::vector<int> result(void);     // return results (one sample as a vector of int)
		double getBestEnergy(void);        // return energy of the best found configuration
} ;

// constructor of IsingModel
// set the size, the type (BINARY or SPIN), and set 2 buffers for configurations
IsingModel::IsingModel(int size, ModelType mtype, std::vector<int>*buf1, std::vector<int>*buf2) {
	isingType= mtype;
	isingSize= size;
	assert(size>0);
	if(isingType == SPIN) defaultModelVal= -1;
	else defaultModelVal= 0;
	// set the configuration buffers
	assert(buf1 && buf2);
	currentConfig= buf1;
	previousConfig= buf2;
}


// set a given initial configuration as random, in accordance to the chosen model
void IsingModel::initRandomConfig(std::vector<int>*configuration) {
	assert(isingSize>0);
	assert(configuration);
	for(int i=0 ; i < isingSize ; i++) {
		if(isingType == BINARY) (*configuration)[i]= uniformDist(2); // 0 or 1
		else (*configuration)[i]= 2*uniformDist(2)-1 ; // -1 or 1
	}
}

// initialize model with h and J, and set the base configuration to start with
void IsingModel::initModel(std::vector<double>*J, std::vector<double>*h) {
	// verify the instance exists
	assert(currentConfig);
	// set J and h
	myJ= J;
	myh= h;
	// now we can start the model with configuration (random) and associated energy
	initRandomConfig(currentConfig);
	currentEnergy= getEnergy(currentConfig);
}

// flip a spin value according to the model
int IsingModel::flip(int val) {
	int ret= val;
	// case spin: change the sign
	if(isingType == SPIN) {
		ret= -ret; 
		assert(abs(ret)== 1); // verify -1 or +1
	}
	// case BINARY: flip between 0 and 1
	else { // BINARY case normally
		ret= 1- ret;
		assert( (ret==0) || (ret == 1) ); // verify 0 or 1 only
	}
	return ret;
}

// generate a new candidate configuration
void IsingModel::generateCandidateConfiguration(void) {
	// select randomly one bit to flip
	int toflip= uniformDist(isingSize);
	assert(previousConfig!= nullptr && currentConfig!= nullptr);
	// copy new configuration from old while flipping the selected bit
	for(int i=0 ; i < isingSize ; i++) if(i != toflip) (*previousConfig)[i]= (*currentConfig)[i];
		else (*previousConfig)[i]= flip((*currentConfig)[i]);
	// then exchange previous and current pointer for current and previous configuration
	std::swap(previousConfig, currentConfig);
	// and finally update energy values
	previousEnergy= currentEnergy;
	currentEnergy= getEnergy(currentConfig);
#ifdef __DEBUG__
	for(int i= 0 ; i < isingSize ; i++) {std::cout<<(*currentConfig)[i];}; std::cout<<"= "<<currentEnergy<<std::endl;
#endif
}

// calculate energy of a configuration
double IsingModel::getEnergy(std::vector<int> *isingConfiguration) {
	double energy= 0.0 ;
	
	assert(myJ && myh);
	// it will work regardless of BINARY or SPIN configuration because c_i^2=c_i for binary case
	// E= \sum_i h * c_i + \sum_{i,j} J_{i,j} c_i c_j 
	for(int i_index= 0; i_index< isingSize ; i_index++) {
		energy+= (*myh)[i_index]*((double)(*isingConfiguration)[i_index]); // works for SPIN or BINARY
		
		for(int j_index= 0; j_index < isingSize ; j_index++) energy+= (*myJ)[i_index*isingSize+j_index]* (*isingConfiguration)[i_index]*(*isingConfiguration)[j_index];
	}
	return energy;
}

// return the energy of the current trial configuration
double IsingModel::getCurrentEnergy(void) {
	return currentEnergy;
}

// return the energy associated with the previous trial configuration
double IsingModel::getPreviousEnergy(void) {
	return previousEnergy;
}

// revert to the previous configuration (case of unfavorable configuration)
void IsingModel::discardCurrent(void) {
	std::swap(currentConfig, previousConfig);
	std::swap(currentEnergy, previousEnergy);
}

std::vector<int>* IsingModel::getCurrentConfig(void) {
	return currentConfig;
}


// Class SimulatedAnnealing
// implementation


// constructor: set the size of the model and its type (BINARY or SPIN)
SimulatedAnnealing::SimulatedAnnealing(int msize, ModelType mtype) {
	assert(msize>0);
	Isize= msize;
	//std::vector<int>*configbuf= new std::vector<int>*[2];
	configbuf[0].resize(msize, 0);
	configbuf[1].resize(msize,0);
	configbuf[2].resize(msize,0);  // configbuf[2] is the best found configuration
	//configbuf[0]= new std::vector<int>[msize];
	//configbuf[1]= new std::vector<int>[msize];
	//std::vector<int>*configbuf2= new std::vector<int>[msize];
	myIsing= new IsingModel(msize, mtype, &configbuf[0], &configbuf[1]);
	setCurrentAsBest();
}

// Set values of matrix J and vector h
void SimulatedAnnealing::initCoupling(std::vector<double>*J, std::vector<double>*h) {
	assert(myIsing);
	myIsing->initModel(J,h); // set vector h and matrix J in the associated Ising model
}

// set initial temperature
void SimulatedAnnealing::setInitialTemperature(double temp) {
	initTemperature= temp; // set initial temperature of the heuristic
}

// set number of swipes per level of temperature
void SimulatedAnnealing::setSwipeNumber(int nswipe) {
	numSwipe= nswipe; // set swipe number
}

void SimulatedAnnealing::setScale(double setscale) {scale= setscale;}

void SimulatedAnnealing::setCurrentAsBest(void) {
	configbuf[2]= *(myIsing->getCurrentConfig()); // copy current configuration of Ising model
	bestEnergy= myIsing->getCurrentEnergy(); // and its energy
}

double SimulatedAnnealing::getBestEnergy(void) { return bestEnergy;}


void SimulatedAnnealing::run(void) {
	// Decrease exponentially the temperature starting from initTemperature
  	for (double currentTemperature = initTemperature;
       	currentTemperature > endingTemperature;
       	currentTemperature *= deltaTemperature) {
       	assert(deltaTemperature>0.0 && deltaTemperature < 1.0);
       	// For each temperature level try numSwipe random configurations (swipes)
       	for(int i= 0 ; i < numSwipe ; i++) {
       		// get a new candidate configuration
            	myIsing->generateCandidateConfiguration();
                        //trialConfig.printMatrix(); // debug purpose
            	// get energy difference 
            	double delta = myIsing->getCurrentEnergy()-myIsing->getPreviousEnergy(); 
            	// probilistic selection of the configuration
            	if (uniform() < std::exp(-scale * delta / currentTemperature))
                     	{  }   // bestConfig = trialConfig; (implicitly done so nothing to do)
            	else myIsing->discardCurrent(); // otherwise discard if not selected
                	//printf("Best="); bestConfig.printMatrix();  // debug purpose
        		if(myIsing->getCurrentEnergy() < bestEnergy)
                		setCurrentAsBest(); // selectCurrent as best configuration
        	}
  	}
}

std::vector<int> SimulatedAnnealing::result(void) {
	//assert(configbuf[2] != nullptr);
	return configbuf[2];
}


// Python3 hooks using CTypes
// TBD

#ifdef __UNITARY_TEST_BINARY_
// Test on G1 case

int main(int argc, char**argv) {
	// define J and h
	std::vector<double> myJ= { 0, 0, 0, 0, +16, +16, 0, 0, 
				0, 0, 0, 0, 0, 0, +16, +16, 
				0, 0, 0, 0, +16, 0, +16, 0, 
				0, 0, 0, 0, 0, +16, 0, +16, 
				0, 0, 0, 0, 0, +16, +16, 0, 
				0, 0, 0, 0, 0, 0, 0, +16, 
				0, 0, 0, 0, 0, 0, 0, +16, 
				0, 0, 0, 0, 0, 0, 0, 0 }; 
	std::vector<double> myh= {-17, -17, -17, -17, -17, -17, -17, -17};
	// instanciate the heuristic in BINARY mode
	SimulatedAnnealing mySim(myh.size(), BINARY);
	
	mySim.setSwipeNumber(myh.size());
	mySim.initCoupling(&myJ, &myh);
	
	mySim.run();
	std::vector<int> myres=mySim.result();
	
	// output results
	std::cout<<"Found solution: (";
	for( int i= 0 ; i < myres.size() ; i++) {
		std::cout<<myres[i]<<" ";
		if (i != myres.size()-1) std::cout<<",";
	}
	std::cout<<") = "<<mySim.getBestEnergy()<<std::endl;
		
	return 0;
}
#endif

#ifdef __UNITARY_TEST_SPIN_
// Test on a simple maxcut problem

int main(int argc, char**argv) {
	// define J and h
	std::vector<double> myJ= { 0, 1, 1, 0, 0,
				0, 0, 0, 1, 0,  
				0, 0, 0, 1, 1,   
				0, 0, 0, 0, 1,  
				0, 0, 0, 0, 0 }; 
	std::vector<double> myh= {0, 0, 0, 0, 0};
	// instanciate heuristic class in SPIN mode
	SimulatedAnnealing mySim(myh.size(), SPIN);
	
	// set parameters
	mySim.setSwipeNumber(myh.size());
	mySim.initCoupling(&myJ, &myh);
	
	// run the heuristic and retrieve the results
	mySim.run();
	std::vector<int> myres=mySim.result();
	
	// output results
	std::cout<<"Found solution: (";
	for( int i= 0 ; i < myres.size() ; i++) {
		std::cout<<myres[i]<<" ";
		if (i != myres.size()-1) std::cout<<",";
	}
	std::cout<<") = "<<mySim.getBestEnergy()<<std::endl;
		
	return 0;
}
#endif

#ifdef __C_TYPES_LIB_
extern "C" // use C link mode for following function, to allow use of ctypes in Python3
{
	__attribute__((visibility("default"))) 
	// let us define the basic interface function. 
	// It needs the size of vector, its type (BOOLEAN or SPIN), J matrix and h vector,
	// and a returns a vector for the solution that was found
	// it returns also a double which is the energy associated with the candidate solution
	double runSA(int msize, ModelType mtype, double *theJ, double *theh, int *ret) {
		double retEnergy= 0;
		// convert the input J matrix and h vector in std::vector format
    		std::vector<double> vecJ(theJ, theJ + msize*msize);
    		std::vector<double> vech(theh, theh + msize);
    		// instanciate SA
		SimulatedAnnealing mySim(msize, mtype) ;
		
		// set parameters for SA
		mySim.setSwipeNumber(msize);
		mySim.initCoupling(&vecJ, &vech);
		
		// run the heuristic
		mySim.run();
		std::vector<int> result_vec= mySim.result();

    		// Copy the data with sanity check on the size of the return vector (just in case)
		size_t n = std::min(vech.size(), static_cast<size_t>(msize));
		for(size_t i = 0; i < n; ++i) {
		    ret[i] = result_vec[i];
		}
 		// and prepare to return the energy of the best found solution
		retEnergy= mySim.getBestEnergy();
		
		return retEnergy;
	}
	
	// Beyond the simple execution, runSAmt uses openMP to compute several samples in parallel and return them
	// It needs the size of vector, its type (BOOLEAN or SPIN), J matrix and h vector, a number of samples
	// returns a vector of int for the solution that was found and their associated energy in another vector
	void runSAmt(int msize, ModelType mtype, double *theJ, double *theh, int nsamp, double *energies, int *ret) {
		//double retEnergy= 0;
    		std::vector<double> vecJ(theJ, theJ + msize*msize);
    		std::vector<double> vech(theh, theh + msize);
    		
    		//std::cout<<"Max threads="<<omp_get_max_threads()<<std::endl;
		
		#pragma omp parallel for
		for(int si= 0 ; si < nsamp ; si++) {
			// For thread safety, allocate mSim for each iteration
			SimulatedAnnealing mySim(msize, mtype) ;
			// Then set parameters
			mySim.setSwipeNumber(msize); // (SA as O(n) complexity i.e. good compromise)
			mySim.initCoupling(&vecJ, &vech); // coupling matrix and vector
			// run the Simulated Annealing
			mySim.run();
			// copy the results in the ret table
			std::vector<int> result_vec= mySim.result();
    			// Protect against buffer overflow
    			size_t ncopy = std::min(result_vec.size(), static_cast<size_t>(msize));

    			// Copy the data
			for(size_t ci = 0; ci < ncopy; ++ci) {
		    		ret[si*msize+ci] = result_vec[ci];
			}
			// and don't forget the energy of the best found solution
			energies[si]= mySim.getBestEnergy();		
		}
	}
	
}
#endif
