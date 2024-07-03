#!/usr/bin/env nextflow

// create input channel
input_ch = Channel.fromPath(params.input)

// load modules
include { READ_DATA                        } from './modules/reading.nf'
include { RUNMODEL as LINEAR_REGRESSION    } from './modules/runmodel.nf'
include { RUNMODEL as K_NEAREST_NEIGHBOURS } from './modules/runmodel.nf'
include { RUNMODEL as RANDOM_FOREST        } from './modules/runmodel.nf'

// run workflow
workflow {
        READ_DATA( input_ch, "$projectDir/scripts/import.R" )
        LINEAR_REGRESSION( READ_DATA.out.dataset, "$projectDir/scripts/linear_regression.R", "linreg" )
        K_NEAREST_NEIGHBOURS( READ_DATA.out.dataset, "$projectDir/scripts/knn.R", "knn" )
        RANDOM_FOREST( READ_DATA.out.dataset,"$projectDir/scripts/random_forest.R", "rf" )
}
