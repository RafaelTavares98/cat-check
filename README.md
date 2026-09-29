# cat-check

A classifier that knows when to shut up.

Every image classifier answers. Show one a photo of a chair, a blurry
smear or a screenshot of a spreadsheet, and it will tell you, at 94%
confidence, which of its two classes that is. The accuracy on the test
set was never wrong. The test set simply never contained a chair.

```
cat-check judge
cat-check curve
cat-check learn
```

## What it prints

```
Model ................. logistic on 1,257 pictures, handwritten digits, is it a three
Judged ................ 526 of 540 pictures, 97% coverage
Accuracy .............. 98.7% of the ones it judged
Refused ............... 14 pictures
  nothing like it ..... 5, further from the training pictures than the limit
  too unsure .......... 9, the two classes were within the margin
Verdict ............... Trust it inside its coverage, and read the refusals.
Reason ................ 98.7% right across 97% of the pictures. The rest it handed back rather than guessing.
Strangers ............. 99% of 300 pictures that are not digits at all were refused
```

The last line is the one that matters. Three hundred pictures that are
not handwriting at all were put in front of it, and it declined 99% of
them. A classifier with two words in its vocabulary will otherwise use
one of them on everything you show it.

## Accuracy is half a sentence

A model that answers everything at 97.4% and a model that answers 94 in
100 at 99.2% are different products, and one number cannot tell them
apart. So the number here is always **accuracy at a coverage**.

```
cat-check curve
```

```
  margin   coverage   accuracy
     0.0       99%     97.4%
     0.2       97%     98.9%
     0.4       97%     98.9%
     0.6       94%     99.2%
```

Widen the margin and it answers less and is righter. That relationship
is the entire claim of the project, and there is a test that fails if it
ever stops holding. If a wider margin did not raise accuracy, the
confidence number would mean nothing and the refusals would be theatre.

## The two refusals

**Too unsure.** The two classes sit within the margin of each other.
Answering there is a coin flip wearing a percentage.

**Nothing like it.** The picture is further from everything the model
was trained on than 99% of that training data is, measured in the few
dimensions the model actually looks at. This is the one that catches the
chair.

## Learning from being corrected

```
cat-check learn
```

```
Corrections ........... 10 pictures a person relabelled
Before ................ 99.4% at 98% coverage
After ................. 99.4% at 99% coverage
Verdict ............... Keep the new model.
Reason ................ It judges 99% against 98%, and is right 99.4% against 99.4%.
```

The pictures worth learning from are the ones it got wrong and the ones
it refused. A normal training set contains neither, which is why a model
that never collects them stops improving on the day it ships.

Half the mistakes are folded into the training set and the other half
are kept back, so the before and the after are measured on pictures
neither model was trained on. Measuring on the corrections themselves
would show a miracle every time.

The verdict can also be **keep the old model**: when corrections make it
worse, the message says to check who wrote the labels.

## The pictures

`sklearn.datasets.load_digits`: 1,797 handwritten digits at eight by
eight, which ship inside scikit-learn. No download, no key, no network.
The question is "is this a three", and threes are confused with fives,
eights and nines often enough to make it a real question.

The strangers are **invented on purpose**: half are digits with their
pixels shuffled, half are plain noise. Neither is handwriting.

Unseen digits were the first idea and it was wrong. A four really does
look like the things the model was trained on, so refusing it would be
the wrong behaviour, and a 7% refusal rate on unseen digits was the code
telling the truth about a bad test. The strangers had to be pictures
that are genuinely not from this world.

CIFAR-10 was the first plan, cats against everything else. The download
stalled at nothing for three minutes, and a repository that only runs
on a good connection is a repository most people cannot run. The name
stayed because the question is the same one.

## Running it

```
pip install -r requirements.txt
python -m cat_check.cli judge
python -m cat_check.cli curve
python -m cat_check.cli learn
```

The exit code is 0 when the model is worth trusting and 1 when it
refuses too much or is wrong too often.

## The tests

```
python -m pytest
```

22 tests, none of which touches a network, a database or a file. The
pictures in them are invented and deliberately overlap: a pair of
classes that separates perfectly makes every probability a 0 or a 1, and
then the abstention tests pass without proving anything. That happened
on the first run and the fixtures were rewritten.

The interesting ones:

* A wider margin must judge less and be righter.
* A stranger must be refused as "nothing like it", and an ordinary
  picture must not be.
* Accuracy is computed only over what was judged.
* The report must never print an accuracy without a coverage beside it,
  and a test reads the output to check.
* Corrections that make the model worse keep the old one.

## What this is not

Not deep learning, not a service, not a labelling tool. One model, two
classes, small pictures.

The out-of-distribution check is a distance to the training data, which
is the simple version. It catches noise and shuffled pixels easily and
would struggle with something subtler, such as the same digits written
by a different hand. Saying so is cheaper than pretending otherwise.
