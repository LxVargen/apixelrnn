from textgenrnn import textgenrnn
textgen = textgenrnn(weights_path='output4test_weights.hdf5',
                       vocab_path='output4test_vocab.json',
                       config_path='output4test_config.json')

textgen.generate_samples(max_gen_length=1000)
