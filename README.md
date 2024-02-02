Ibrahim Najmudin

This is a practice repository intended to teach me about structuring projects.

It will be for my MPhys project : Laser Amplification Of Incoherent Sources.

-> Function folders descriptions

    -> common
        Stores functions which are generally used in initialising inputs (x-section data + intensity profiles + etc.)

    -> resources
        Stores x-section data
    
    -> runs
        Stores geometries for laser amplification, e.g. Orion, Amica.

    -> test
        Stores the main scripts which test the geometries from the runs folder
        Typically produce output plots or output energies.
        Also typically programmed in the iteration method in them

    -> unused
        Stores old versions of tests scripts

-> params.py stores all global parameters.

