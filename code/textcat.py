#!/usr/bin/env python3
"""
Computes the total log probability of the sequences of tokens in each file,
according to a given smoothed trigram model.  
"""
import argparse
import logging
import math
from pathlib import Path
import torch

from probs import Wordtype, LanguageModel, num_tokens, read_trigrams

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model1",
        type=Path,
        help="path to the trained model",
    )
    parser.add_argument(
        "model2",
        type=Path,
        help="path to the trained model",
    )
    parser.add_argument(
        "prior",
        type=float,
        help="prior probability for the language model",
    )
    parser.add_argument(
        "test_files",
        type=Path,
        nargs="*"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=['cpu','cuda','mps'],
        help="device to use for PyTorch (cpu or cuda, or mps if you are on a mac)"
    )


    # for verbosity of logging
    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", dest="logging_level", action="store_const", const=logging.DEBUG
    )
    verbosity.add_argument(
        "-q", "--quiet",   dest="logging_level", action="store_const", const=logging.WARNING
    )

    return parser.parse_args()

def catagorize(lm1: LanguageModel, lm2: LanguageModel, prior: float, file: Path) -> int:
    """ The function takes in 2 language models, one for each catagory, 
        and prior for the first language model, and a file. Using the log probability of 
        the file given, classify the catagory of the file using log probability of
        the file.
    """
    log_prob1 = file_log_prob(file, lm1)
    log_prob2 = file_log_prob(file, lm2)

    log_prior1 = math.log(prior)
    log_prior2 = math.log(1 - prior)

    log_posterior1 = log_prob1 + log_prior1
    log_posterior2 = log_prob2 + log_prior2

    if log_posterior1 > log_posterior2:
        return 1
    else:
        return 2


def file_log_prob(file: Path, lm: LanguageModel) -> float:
    """The file contains one sentence per line. Return the total
    log-probability of all these sentences, under the given language model.
    (This is a natural log, as for all our internal computations.)
    """
    log_prob = 0.0

    x: Wordtype; y: Wordtype; z: Wordtype    # type annotation for loop variables below
    for (x, y, z) in read_trigrams(file, lm.vocab):
        log_prob += lm.log_prob(x, y, z)  # log p(z | xy)

        # If the factor p(z | xy) = 0, then it will drive our cumulative file 
        # probability to 0 and our cumulative log_prob to -infinity.  In 
        # this case we can stop early, since the file probability will stay 
        # at 0 regardless of the remaining tokens.
        if log_prob == -math.inf: break 

        # Why did we bother stopping early?  It could occasionally
        # give a tiny speedup, but there is a more subtle reason -- it
        # avoids a ZeroDivisionError exception in the unsmoothed case.
        # If xyz has never been seen, then perhaps yz hasn't either,
        # in which case p(next token | yz) will be 0/0 if unsmoothed.
        # We can avoid having Python attempt 0/0 by stopping early.
        # (Conceptually, 0/0 is an indeterminate quantity that could
        # have any value, and clearly its value doesn't matter here
        # since we'd just be multiplying it by 0.)

    return log_prob



def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)
    # print(f'prior = {args.prior}\nmodel1 = {args.model1}\nmodel1 = {args.model2}')

    # Specify hardware device where all tensors should be computed and
    # stored.  This will give errors unless you have such a device
    # (e.g., 'gpu' will work in a Kaggle Notebook where you have
    # turned on GPU acceleration).
    if args.device == 'mps':
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                logging.critical("MPS not available because the current PyTorch install was not "
                    "built with MPS enabled.")
            else:
                logging.critical("MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine.")
            exit(1)
    torch.set_default_device(args.device)
        
    log.info("Testing...")
    lm1 = LanguageModel.load(args.model1, device=args.device)
    lm2 = LanguageModel.load(args.model2, device=args.device)
    prior = args.prior
    
    if lm1.vocab != lm2.vocab:
        raise ValueError("Vocab Files do not match")
    # We use natural log for our internal computations and that's
    # the kind of log-probability that file_log_prob returns.
    # We'll print that first.

    log.info("Per-file Classification:")
    nMod1 = 0
    nMod2 = 0
    correct = 0
    total = 0
    total_log_prob1 = 0.0
    total_log_prob2 = 0.0
    for file in args.test_files:
        current_file = Path(file)
        file_name = current_file.name
        log_prob1: float = file_log_prob(file, lm1)
        log_prob2: float = file_log_prob(file, lm2)
        # print(f"{log_prob:g}\t{file}")
        total_log_prob1 += log_prob1
        total_log_prob2 += log_prob2
        classify: int = catagorize(lm1, lm2, prior, file)
        actual: int = 1 if args.model1.stem in file_name else 2
        correct += 1 if classify == actual else 0
        total += 1
        nMod1, nMod2 = (nMod1 + 1, nMod2) if classify == 1 else (nMod1, nMod2 + 1)
        print(f'{args.model1}\t{file}') if classify == 1 else print(f'{args.model2}\t{file}')
    print(f'{nMod1} files were more probably from {args.model1} ({nMod1/(nMod1+nMod2)})')
    print(f'{nMod2} files were more probably from {args.model2} ({nMod2/(nMod1+nMod2)})')
    print(f'total error rate = {(total - correct)/total}')

    # But cross-entropy is conventionally measured in bits: so when it's
    # time to print cross-entropy, we convert log base e to log base 2, 
    # by dividing by log(2).

    bits_mod1 = -total_log_prob1 / math.log(2)   # convert to bits of surprisal
    bits_mod2 = -total_log_prob2 / math.log(2)   # convert to bits of surprisal

    # We also divide by the # of tokens (including EOS tokens) to get
    # bits per token.  (The division happens within the print statement.)

    tokens = sum(num_tokens(test_file) for test_file in args.test_files)
    print(f"Overall cross-entropy for {args.model1}:\t{bits_mod1 / tokens:.5f} bits per token")
    print(f"Overall cross-entropy for {args.model2}:\t{bits_mod2 / tokens:.5f} bits per token")


if __name__ == "__main__":
    main()

