use clap::Clap;
use condor::modules::{NNModule, LanguageModel};
use condor::utils::{ExponentialAverage, count_parameters, readable_number, train_progress_bar, test_progress_bar, sample_1d};
use dataflow::dataloader::Dataloader;
use dataflow::pipeline::{RandomLoader, Stateful, Node, Sort, Batch};
use dataflow::tokenization::{WordpieceTokenizer, Tokenizer};
use dataflow::vocab::{Vocab, WordPieceVocab};
use tch::IndexOp;
use tch::{Device, Kind, Tensor, nn::{self, AdamW, Optimizer, OptimizerConfig}};
use text_io::read;

const BATCH_SIZE: usize = 10;
const BATCH_AGGREGATIONS: usize = 5;
const LEARNING_RATE: f64 = 0.0001;
const EPOCHS: usize = 10;
const LAYERS: i64 = 12;
const HEADS: i64 = 16;
const EMBED_SIZE: i64 = 512;
const DROPOUT: f64 = 0.1;

fn main() {
    println!("|TRANSFORMER LANGUANGE MODEL|");
    let args = Args::parse();
    
    if args.test {
        test();
    } else {
        train(args.load);
    }
}

fn train(load: bool) {
    // Create dataset
    println!("Builing Dataset...");
    let pipeline = RandomLoader::new(vec!["/home/jafioti/Datasets/wiki/lm_dataset.txt".to_string()])
        .add_node(Stateful::new(
            |strings: Vec<String>, (tokenizer, vocab)| {
                let strings = strings.into_iter().map(|s| s.replace("\\", "").replace("/", "").replace('"', "")).collect();
                let tokens = tokenizer.batch_tokenize(strings);
                let indexes = vocab.batch_indexes_from_tokens(&tokens).unwrap();
                indexes.into_iter().map(|indexes| {
                    indexes[..64.min(indexes.len())].iter().map(|i| *i as i32).collect()
                }).collect()
            },
            (dataflow::tokenization::WordpieceTokenizer::load(), dataflow::vocab::WordPieceVocab::load())
        )).add_node(Sort::new(
            |a: &Vec<i32>, b| a.len().cmp(&b.len())
        )).add_node(Batch::new(BATCH_SIZE))
        .add_fn(|indexes: Vec<Vec<Vec<i32>>>| {
            indexes.into_iter().map(|batch| {
                let (mut inputs, mut targets): (Vec<Vec<i32>>, Vec<Vec<i32>>) = batch.into_iter()
                    .map(|indexes| (indexes[..indexes.len() - 1].to_vec(), indexes[1..].to_vec())).unzip();
                let max_inputs = inputs.iter().map(|i| i.len()).max().unwrap();
                inputs.iter_mut().for_each(|i| i.extend(vec![0; max_inputs - i.len()].iter()));
                let max_targets = targets.iter().map(|i| i.len()).max().unwrap();
                targets.iter_mut().for_each(|i| i.extend(vec![0; max_targets - i.len()].iter()));
                (Tensor::of_slice2(&inputs), Tensor::of_slice2(&targets))
            }).collect()
        });
    let mut train_dataset = Dataloader::new(pipeline)
        .load_block_size(10_000);
    let pipeline = RandomLoader::new(vec!["/home/jafioti/Datasets/wiki/lm_valid_dataset.txt".to_string()])
        .add_node(Stateful::new(
            |strings: Vec<String>, (tokenizer, vocab)| {
                let strings = strings.into_iter().map(|s| s.replace("\\", "").replace("/", "").replace('"', "")).collect();
                let tokens = tokenizer.batch_tokenize(strings);
                let indexes = vocab.batch_indexes_from_tokens(&tokens).unwrap();
                indexes.into_iter().map(|indexes| {
                    indexes[..64.min(indexes.len())].iter().map(|i| *i as i32).collect()
                }).collect()
            },
            (dataflow::tokenization::WordpieceTokenizer::load(), dataflow::vocab::WordPieceVocab::load())
        )).add_node(Sort::new(
            |a: &Vec<i32>, b| a.len().cmp(&b.len())
        )).add_node(Batch::new(BATCH_SIZE))
        .add_fn(|indexes: Vec<Vec<Vec<i32>>>| {
            indexes.into_iter().map(|batch| {
                let (mut inputs, mut targets): (Vec<Vec<i32>>, Vec<Vec<i32>>) = batch.into_iter()
                    .map(|indexes| (indexes[..indexes.len() - 1].to_vec(), indexes[1..].to_vec())).unzip();
                let max_inputs = inputs.iter().map(|i| i.len()).max().unwrap();
                inputs.iter_mut().for_each(|i| i.extend(vec![0; max_inputs - i.len()].iter()));
                let max_targets = targets.iter().map(|i| i.len()).max().unwrap();
                targets.iter_mut().for_each(|i| i.extend(vec![0; max_targets - i.len()].iter()));
                (Tensor::of_slice2(&inputs), Tensor::of_slice2(&targets))
            }).collect()
        });
    let mut test_dataset = Dataloader::new(pipeline)
        .load_block_size(10_000);

    
    println!("Training Examples: {}", train_dataset.len());
    println!("Testing Examples: {}", test_dataset.len());

    println!("Building Model...");
    let local_vocab = WordPieceVocab::load();
    let device = Device::cuda_if_available();
    let mut vs = nn::VarStore::new(device);
    let mut model = LanguageModel::new(&vs.root(), EMBED_SIZE, HEADS, LAYERS, local_vocab.len() as i64, condor::modules::PositionalEncoding::Learned, 1000, DROPOUT);
    if load && vs.load("model.pt").is_err() {
        println!("Failed to load model!");
    }
    let mut opt = nn::AdamW::default().build(&vs, LEARNING_RATE).expect("Failed to build optimizer");
    println!("Model Parameters: {}", readable_number(count_parameters(&vs) as i64));

    let tokenizer = dataflow::tokenization::WordpieceTokenizer::load();
    let mut best_loss = f64::MAX;
    for epoch in 0..EPOCHS {
        println!("\nEpoch {}", epoch + 1);
        println!("Training...");
        let (train_loss, train_acc) = train_epoch(&mut model, &mut train_dataset, &mut opt);
        println!("Train PPL: {} Train Acc: {}", train_loss, train_acc);
        println!("Testing...");
        let (test_loss, test_acc) = test_epoch(&mut model, &mut test_dataset);
        println!("Test PPL: {} Test Acc: {}", test_loss, test_acc);

        let sentence = "I swung the ".to_string();
        println!("\nEval Starting Sentence: {}\nGenerated Sentence: {}\n", sentence.clone(), generate(sentence, &mut model, 30, 0.4, &local_vocab, &tokenizer));

        // Save
        if test_loss < best_loss {
            println!("Saving...");
            vs.save("model.pt").expect("Failed to save model");
            best_loss = test_loss;
        }
    }
}

fn test() {
    // Load model
    println!("Loading Model...");
    let local_vocab = WordPieceVocab::load();
    let tokenizer = WordpieceTokenizer::load();
    let device = Device::cuda_if_available();
    let mut vs = nn::VarStore::new(device);
    let mut model = LanguageModel::new(&(&vs.root() / "lm"), EMBED_SIZE, HEADS, LAYERS, local_vocab.len() as i64, condor::modules::PositionalEncoding::Learned, 1000, DROPOUT);
    vs.load("model.pt").expect("Failed to load model parameters!");
    println!("Model Parameters: {}", readable_number(count_parameters(&vs) as i64));
    println!("\n----------------------------------------------");

    loop {
        println!("Input:");
        // Get input
        let input: String = read!("{}\n");
        // Generate tokens
        println!("Generated Sentence: {}", generate(input, &mut model, 20, 0.4, &local_vocab, &tokenizer));
    }
}

fn train_epoch<N: Node<Input = (), Output = (Tensor, Tensor)> + Send + 'static>(model: &mut LanguageModel, dataset: &mut Dataloader<N>, optimizer: &mut Optimizer<AdamW>) -> (f64, f64) 
where N::Output: Send {
    model.train();
    let bar = train_progress_bar((dataset.len() / BATCH_SIZE) as u64);
    let mut loss_avg = ExponentialAverage::new();
    let mut accuracies: Vec<f64> = vec![];
    
    for (i, (mut inputs, mut targets)) in dataset.enumerate() {
        optimizer.zero_grad();
        // Process batch into tensors
        inputs = inputs.to_device(Device::cuda_if_available());
        targets = targets.to_device(Device::cuda_if_available());
        let (batch_size, seq_length) = inputs.size2().unwrap();
        // Run through model
        let output = model.forward(&inputs);

        // Get loss
        let loss = output.view([batch_size * seq_length, output.size()[2]])
            .cross_entropy_for_logits(&targets.view([batch_size * seq_length]).to_kind(Kind::Int64));
        loss.backward();
        loss_avg.update(f64::from(loss).exp());
        if i % BATCH_AGGREGATIONS == 0 {
            optimizer.step();
            optimizer.zero_grad();
        }

        // Get accuracy
        accuracies.push(f64::from(output.accuracy_for_logits(&targets.to_kind(Kind::Int64))));

        bar.inc(1);
        bar.set_message(format!("PPL: {:.2}", loss_avg.value));
    }
    (loss_avg.value, accuracies.iter().sum::<f64>() / accuracies.len() as f64)
}

fn test_epoch<N: Node<Input = (), Output = (Tensor, Tensor)> + Send + 'static>(model: &mut LanguageModel, dataset: &mut Dataloader<N>) -> (f64, f64)
where N::Output: Send {
    model.eval();
    let mut losses = vec![];
    let mut accuracies: Vec<f64> = vec![];
    let bar = test_progress_bar((dataset.len() / BATCH_SIZE) as u64);
    for (mut inputs, mut targets) in dataset {
        // Process batch into tensors
        let (batch_size, seq_length) = inputs.size2().unwrap();
        inputs = inputs.to_device(Device::cuda_if_available());
        targets = targets.to_device(Device::cuda_if_available());
        // Run through model
        let output = model.forward(&inputs);

        // Get loss
        let loss = output.view([batch_size * seq_length, output.size()[2]])
            .cross_entropy_for_logits(&targets.view([batch_size * seq_length]).to_kind(Kind::Int64));
        losses.push(f64::from(loss).exp());

        // Get accuracy
        accuracies.push(f64::from(output.accuracy_for_logits(&targets.to_kind(Kind::Int64))));
        bar.inc(1);
    }
    (losses.iter().sum::<f64>() / losses.len() as f64, accuracies.iter().sum::<f64>() / accuracies.len() as f64)
}

fn generate<T: Tokenizer, V: Vocab>(sentence: String, model: &mut LanguageModel, tokens_to_generate: usize, temperature: f64, local_vocab: &V, tokenizer: &T) -> String {
    model.eval();
    let mut tokens = tokenizer.tokenize(&sentence);
    for _ in 0..tokens_to_generate {
        // Generate new token
        // Vectorize sentence
        let vectorized: Vec<i32> = local_vocab.indexes_from_tokens(&tokens).unwrap()
            .iter().map(|i| {*i as i32}).collect();
        let input = Tensor::of_slice(&vectorized).unsqueeze(0).to(Device::cuda_if_available());
        let output = model.forward(&input);
        let index = sample_1d(output.squeeze_dim(0).i(-1), temperature);
        let token = local_vocab.tokens_from_indexes(&[index]).unwrap().pop().unwrap();
        tokens.push(token);
    }
    tokenizer.untokenize(tokens)
}

/// Basic language model training
#[derive(Clap, Debug)]
#[clap(name = "Transformer Language Model")]
struct Args {
    /// Whether or not to attempt to load the model before training
    #[clap(short, long)]
    load: bool,

    /// Whether or not to run testing instead of training
    #[clap(short, long)]
    test: bool
}