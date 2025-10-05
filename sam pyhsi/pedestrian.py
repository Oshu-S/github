import math
import numpy as np
from sympy import *


class Pedestrian:
    """
    Base Class for creating a Pedestrian
    """

    populationProperties = {}
    meanLognormalModel = 4.28  # mM
    sdLognormalModel = 0.21  # sM

    detK = 14110
    detVelocity = 1.25

    synchedPace = 0
    synchedPhase = 0

    def __init__(self, mass, damp, stiff, pace, phase, location, velocity, iSync):
        """
        this function introduce the properties when creating one pedestrian

        Parameters
        ----------
        mass: human mass
        damp : damping effect of pedesteian
        stiff : stiffness of humans
        pace : pacing frequency
        phase : phase angle
        location : location of mass
        velocity : velocity of travelling mass
        iSync : synchronization

        Returns
        -------
        None.

        """
        self.mass = mass
        self.damp = damp
        self.stiff = stiff
        self.pace = pace
        self.phase = phase
        self.location = location
        self.velocity = velocity
        self.iSync = iSync


def calcPedForce(self, t):
        # Question: What are all the commented out parts in matlab ped_force
        g = 9.81

        W = self.mass * g
        #x = self.location + self.velocity * t  # Position of Pedestrian at each time t

        # Young
        eta = np.array([0.41 * (self.pace - 0.95),
                        0.069 + 0.0056 * self.pace,
                        0.033 + 0.0064 * self.pace,
                        0.013 + 0.0065 * self.pace])
        phi = np.zeros(4)

        # Now assemble final force, and include weight
        N = len(eta)  # No. of additional terms in harmonic series
        F0 = W * np.insert(eta, 0, 1)  # Force amplitudes (constant amplitude for 1)
        beta = 2 * math.pi * self.pace * np.array([i for i in range(N + 1)])  # Frequencies
        phi = np.insert(phi, 0, 0) + self.phase  # Phases - enforce first phase as zero phase
       
        beta = beta[:, np.newaxis]  # Reshape to (N+1, 1)
        phi = phi[:, np.newaxis]    # Reshape to (N+1, 1)
        
        omega = beta * t + phi
        Ft = np.sum(F0[:, np.newaxis] * np.cos(omega), axis=0)
        #Ft = sum(F0 * np.cos(omega))

        return  Ft
    # endregion

class Crowd:

    populationProperties = {}
    """
    an empty dictionary is initialized to store population properties which will be introduced in the following 
    lines of code.  
    """

    def __init__(self, numPedestrians, length, width, sync):
        """
        initialization takes arguments numPedestrians, length, width and sync. Then set the corresponding attributes
        """
        # self.density = density
        self.numPedestrians = numPedestrians
        self.length = length
        self.width = width
        self.sync = sync

        self.area = self.length * self.width
        # self.numPedestrians = int(self.density * self.area)
        self.lamda = self.numPedestrians / self.length

        self.locations = []
        self.iSync = []
        self.pedestrians = []

        # Crowd synchronization
        self.determineCrowdSynchronisation()

    def determineCrowdSynchronisation(self):
        sync = self.sync/100
        self.iSync = np.random.choice([0, 1], size=self.numPedestrians, p=[1 - sync, sync])
        pace = np.random.normal(loc=self.populationProperties['meanPace'], scale=self.populationProperties['sdPace'])
        phase = (2 * math.pi) * (np.random.rand())
        Pedestrian.setPaceAndPhase(pace, phase)

    def addRandomPedestrian(self, location, synched):
        self.pedestrians.append(Pedestrian.randomPedestrian(location, synched))

    def addDeterministicPedestrian(self, location, synched):
        self.pedestrians.append(Pedestrian.deterministicPedestrian(location, synched))

    def addExactPedestrian(self, location, synched):
        """
        Temporary, for testing
        """
        self.pedestrians.append(Pedestrian.exactPedestrian(location, synched))

class SinglePedestrian(Pedestrian):    #update this to run on body damping and frequancy
    """
    Sub Class of Pedestrian
    """
    def __init__(self):
        """
        super().__init__(parameters)
            inherits the parameters from the Base class Pedestrian.

        reintroduce the parameters from Pedestrian Class which overrides the ones set under the parent class

        introduce two other Parameters
        numPedestrians
        """
        
        k = 14.11e3  

        pMass = self.populationProperties['meanMass']
        pDamp = self.populationProperties['meanDamping'] * 2 * math.sqrt(k * pMass)
        pStiff = k
        pPace = 2
        pPhase = 0
        pLocation = 0
        pVelocity = 1.25
        iSync = 0
        super().__init__(pMass, pDamp, pStiff, pPace, pPhase, pLocation, pVelocity, iSync)
        self.numPedestrians = 1
        self.pedestrians = [self] #???

    @classmethod
    def fromDict(cls, crowdOptions):
        return cls()


class DeterministicCrowd(Crowd):

    arrivalGap = 1      

    def __init__(self, numPedestrians, length, width, sync):
        super().__init__(numPedestrians, length, width, sync)
        self.generateLocations()
        self.populateCrowd()

    def generateLocations(self):
        self.locations = -self.arrivalGap*np.array(range(self.numPedestrians))

    def populateCrowd(self):
        for i in range(self.numPedestrians):
            self.addDeterministicPedestrian(self.locations[i], self.iSync[i])

    @classmethod
    def setArrivalGap(cls, arrivalGap):
        cls.arrivalGap = arrivalGap


class RandomCrowd(Crowd):
    def __init__(self, numPedestrians, length, width, sync):
        super().__init__(numPedestrians, length, width, sync)
        self.generateLocations()
        self.populateCrowd()

    def generateLocations(self):
        gaps = np.random.exponential(1 / self.lamda, size=self.numPedestrians)
        self.locations = np.cumsum(gaps, axis=None, dtype=None, out=None)

    def populateCrowd(self):
        for i in range(self.numPedestrians):
            self.addRandomPedestrian(self.locations[i], self.iSync[i])
