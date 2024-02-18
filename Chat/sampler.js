import { Tensor,
    max,
    softmax,
    log_softmax,
    getTopItems} from 'https://cdn.jsdelivr.net/npm/@xenova/transformers';

/**
 * Class that holds a configuration for a generation task.
 * @type {new (kwargs?: GenerationConfigType) => GenerationConfigType}
 */
export const GenerationConfig = /** @type {any} */ (class {

    /**
     * Create a new GenerationConfig object.
     * @param {GenerationConfigType} kwargs 
     */
    constructor(kwargs = {}) {
        // Parameters that control the length of the output
        this.max_length = kwargs.max_length ?? 20;
        this.max_new_tokens = kwargs.max_new_tokens ?? null;
        this.min_length = kwargs.min_length ?? 0;
        this.min_new_tokens = kwargs.min_new_tokens ?? null;
        this.early_stopping = kwargs.early_stopping ?? false;
        this.max_time = kwargs.max_time ?? null;

        // Parameters that control the generation strategy used
        this.do_sample = kwargs.do_sample ?? false;
        this.num_beams = kwargs.num_beams ?? 1;
        this.num_beam_groups = kwargs.num_beam_groups ?? 1;
        this.penalty_alpha = kwargs.penalty_alpha ?? null;
        this.use_cache = kwargs.use_cache ?? true;

        // Parameters for manipulation of the model output logits
        this.temperature = kwargs.temperature ?? 1.0;
        this.top_k = kwargs.top_k ?? 50;
        this.top_p = kwargs.top_p ?? 1.0;
        this.typical_p = kwargs.typical_p ?? 1.0;
        this.epsilon_cutoff = kwargs.epsilon_cutoff ?? 0.0;
        this.eta_cutoff = kwargs.eta_cutoff ?? 0.0;
        this.diversity_penalty = kwargs.diversity_penalty ?? 0.0;
        this.repetition_penalty = kwargs.repetition_penalty ?? 1.0;
        this.encoder_repetition_penalty = kwargs.encoder_repetition_penalty ?? 1.0;
        this.length_penalty = kwargs.length_penalty ?? 1.0;
        this.no_repeat_ngram_size = kwargs.no_repeat_ngram_size ?? 0;
        this.bad_words_ids = kwargs.bad_words_ids ?? null;
        this.force_words_ids = kwargs.force_words_ids ?? null;
        this.renormalize_logits = kwargs.renormalize_logits ?? false;
        this.constraints = kwargs.constraints ?? null;
        this.forced_bos_token_id = kwargs.forced_bos_token_id ?? null;
        this.forced_eos_token_id = kwargs.forced_eos_token_id ?? null;
        this.remove_invalid_values = kwargs.remove_invalid_values ?? false;
        this.exponential_decay_length_penalty = kwargs.exponential_decay_length_penalty ?? null;
        this.suppress_tokens = kwargs.suppress_tokens ?? null;
        this.begin_suppress_tokens = kwargs.begin_suppress_tokens ?? null;
        this.forced_decoder_ids = kwargs.forced_decoder_ids ?? null;

        // Parameters that define the output variables of `generate`
        this.num_return_sequences = kwargs.num_return_sequences ?? 1;
        this.output_attentions = kwargs.output_attentions ?? false;
        this.output_hidden_states = kwargs.output_hidden_states ?? false;
        this.output_scores = kwargs.output_scores ?? false;
        this.return_dict_in_generate = kwargs.return_dict_in_generate ?? false;

        // Special tokens that can be used at generation time
        this.pad_token_id = kwargs.pad_token_id ?? null;
        this.bos_token_id = kwargs.bos_token_id ?? null;
        this.eos_token_id = kwargs.eos_token_id ?? null;

        // Generation parameters exclusive to encoder-decoder models
        this.encoder_no_repeat_ngram_size = kwargs.encoder_no_repeat_ngram_size ?? 0;
        this.decoder_start_token_id = kwargs.decoder_start_token_id ?? null;

        // Wild card
        this.generation_kwargs = kwargs.generation_kwargs ?? {};
    }
});

/**
 * Sampler is a base class for all sampling methods used for text generation.
 */
export class Sampler {
    /**
     * Creates a new Sampler object with the specified generation config.
     * @param {GenerationConfigType} generation_config The generation config.
     */
    constructor(generation_config) {
        this.generation_config = generation_config;
    }

    /**
     * Executes the sampler, using the specified logits.
     * @param {Tensor} logits
     * @param {number} index
     * @returns {void}
     */
    _call(logits, index = -1) {
        // Sample from logits, of dims [batch, sequence_length, vocab_size].
        // If index is specified, sample from [batch, index, vocab_size].
        return this.sample(logits, index);
    }

    /**
     * Abstract method for sampling the logits.
     * @param {Tensor} logits
     * @param {number} index
     * @throws {Error}
     */
    sample(logits, index) {
        throw Error("sample should be implemented in subclasses.")
    }

    /**
     * Returns the specified logits as an array, with temperature applied.
     * @param {Tensor} logits
     * @param {number} index
     * @returns {Float32Array}
     */
    getLogits(logits, index) {
        let vocabSize = logits.dims.at(-1);

        let logs = /** @type {Float32Array} */(logits.data);

        if (index === -1) {
            logs = logs.slice(-vocabSize);
        } else {
            let startIndex = index * vocabSize;
            logs = logs.slice(startIndex, startIndex + vocabSize);
        }

        // add temperature
        if (this.generation_config.temperature > 0) {
            logs = logs.map(x => x / this.generation_config.temperature)
        }
        return logs;
    }

    /**
     * Selects an item randomly based on the specified probabilities.
     * @param {Array} probabilities An array of probabilities to use for selection.
     * @returns {number} The index of the selected item.
     */
    randomSelect(probabilities) {
        // Return index of chosen item
        let sumProbabilities = probabilities.reduce((acc, curr) => acc + curr, 0);

        let r = Math.random() * sumProbabilities;
        for (let i = 0; i < probabilities.length; ++i) {
            r -= probabilities[i];
            if (r <= 0) {
                return i;
            }
        }
        return 0; // return first (most probable) as a fallback
    }

    /**
     * Returns a Sampler object based on the specified options.
     * @param {GenerationConfigType} generation_config An object containing options for the sampler.
     * @returns {Sampler} A Sampler object.
     */
    static getSampler(generation_config) {
        // - *greedy decoding*: `num_beams=1` and `do_sample=False`
        // - *contrastive search*: `penalty_alpha>0` and `top_k>1`
        // - *multinomial sampling*: `num_beams=1` and `do_sample=True`
        // - *beam-search decoding*: `num_beams>1` and `do_sample=False`
        // - *beam-search multinomial sampling*: `num_beams>1` and `do_sample=True`
        // - *diverse beam-search decoding*: `num_beams>1` and `num_beam_groups>1`
        // - *constrained beam-search decoding*: `constraints!=None` or `force_words_ids!=None`

        // NOTE: beam search is implemented directly into the generation function
        if (generation_config.do_sample) {
            return new MultinomialSampler(generation_config);

        } else if (generation_config.num_beams > 1) {
            return new BeamSearchSampler(generation_config);

        } else {
            if (generation_config.num_return_sequences > 1) {
                throw Error(`num_return_sequences has to be 1 when doing greedy search, but is ${generation_config.num_return_sequences}.`)
            }
            return new GreedySampler(generation_config);
        }
    }
}

/**
 * Class representing a Greedy Sampler.
 * @extends Sampler
 */
class GreedySampler extends Sampler {
    /**
     * Sample the maximum probability of a given logits tensor.
     * @param {Tensor} logits
     * @param {number} [index=-1]
     * @returns {Array} An array with a single tuple, containing the index of the maximum value and a meaningless score (since this is a greedy search).
     */
    sample(logits, index = -1) {
        // NOTE: no need to do log_softmax here since we only take the maximum
        let logs = this.getLogits(logits, index);
        let argmax = max(logs)[1];

        // Note: score is meaningless in this context, since we are performing
        // greedy search (p = 1 => log(p) = 0)
        return [
            [argmax, 0]
        ];
    }
}

/**
 * Class representing a MultinomialSampler.
 * @extends Sampler
 */
class MultinomialSampler extends Sampler {

    /**
     * Sample from the logits.
     * @param {Tensor} logits
     * @param {number} index
     * @returns {Array}
     */
    sample(logits, index = -1) {
        let k = logits.dims.at(-1); // defaults to vocab size
        if (this.generation_config.top_k > 0) {
            k = Math.min(this.generation_config.top_k, k);
        }

        // Get logits of nth token
        const logs = this.getLogits(logits, index);

        // Get top k tokens
        const topLogits = getTopItems(logs, k);

        // Compute softmax over logits
        const probabilities = softmax(topLogits.map(x => x[1]));

        return Array.from({ length: this.generation_config.num_beams }, () => {
            const sampledIndex = this.randomSelect(probabilities);
            return [
                topLogits[sampledIndex][0], // token id
                Math.log(probabilities[sampledIndex]), // score
            ];
        });
    }
}

