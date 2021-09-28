# langauge_model
A simple language model written in Rust

Usage:  
--load: Loads the model from model.pt before training/testing.  
--test: Interactively generate text, no training. Also loads the model.

This repository demonsrates how an entire machine learning project can be written in Rust, using a variety of both in-house and external crates. This project utilizes condor for model architectures, and mako for data processing. When compared to training models of the same size in Python, this repo outperformed them sometimes by a factor of 2, while never runtime erroring. This repo is mostly a proof of concept that it is not only feasible to write ML projects entirely in Rust, but that it is ergonomic as well.
