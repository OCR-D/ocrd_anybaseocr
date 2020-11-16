Change Log
==========
Versioned according to [Semantic Versioning](http://semver.org/).

## Unreleased

Fixed:

  * crop: remove `operation_level` parameter
  * crop: remove `force` parameter
  * crop: fix setting pageId of derived images

## [1.0.2] - 2020-09-24

Fixed:

  * logging according to https://github.com/OCR-D/core/pull/599

## [1.0.1] - 2020-08-24

Fixed:

  * replace copy&pasted metadata addition with `self.add_metadata`, #67

## [1.0.0] - 2020-08-21

Fixed:

  * all processors runnable (though dewarping only with CUDA)
  * fix pix2pixhd installation, #64
  * adapt to 1-output-file-group convention, #66

## [0.0.5] - 2020-08-04

Fixed:

  * Refactoring to make tools installable, with `--help` and `-J` working

## [0.0.4] - 2020-07-08

Fixed:

  * keras should be `>= 2.3.0, < 2.4.0`, #63

## [0.0.3] - 2020-05-14

Fixed:

  * Graceful degradation to CPU processing if CUDA not available - AGAIN, ht @bertsky, #58

## [0.0.2] - 2020-05-06

Fixed:

  * Graceful degradation to CPU processing if CUDA not available, #56

<!-- link-labels -->
[1.0.2]: ../../compare/v1.0.2...v1.0.1
[1.0.1]: ../../compare/v1.0.1...v1.0.0
[1.0.0]: ../../compare/v1.0.0...v0.0.5
[0.0.5]: ../../compare/v0.0.5...v0.0.4
[0.0.4]: ../../compare/v0.0.3...v0.0.4
[0.0.3]: ../../compare/v0.0.2...v0.0.3
[0.0.2]: ../../compare/HEAD...v0.0.2
