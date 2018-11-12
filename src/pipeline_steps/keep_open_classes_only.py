import nltk
from nltk import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin


class KeepOpenClassesOnly(BaseEstimator, TransformerMixin):
    """
    Categories that will usually be open classes:
    - adjectives
    - adverbs
    - nouns
    - verbs (except auxiliary verbs)
    - interjections

    Keep Open Classes or things that contains at least some of Open Classes:
    - adjectives:
    JJ: adjective or numeral, ordinal
    JJR: adjective, comparative
    JJS: adjective, superlative
    - adverbs:
    RB: adverb
    RBR: adverb, comparative
    RBS: adverb, superlative
    WRB: Wh-adverb
    - nouns:
    NN: noun, common, singular or mass
    NNP: noun, proper, singular
    NNPS: noun, proper, plural
    NNS: noun, common, plural
    - verbs (except auxiliary verbs):
    VB: verb, base form
    VBD: verb, past tense
    VBG: verb, present participle or gerund
    VBN: verb, past participle
    VBP: verb, present tense, not 3rd person singular
    VBZ: verb, present tense, 3rd person singular
    - interjections:
    UH: interjection

    Other words that are not in "Open Classes":
    CC: conjunction, coordinating
    CD: numeral, cardinal
    DT: determiner
    EX: existential there
    FW: foreign word
    IN: preposition or conjunction, subordinating
    LS: list item marker
    MD: modal auxiliary
    PDT: pre-determiner
    POS: genitive marker
    PRP: pronoun, personal
    PRP$: pronoun, possessive
    RP: particle
    SYM: symbol
    TO: "to" as preposition or infinitive marker
    WDT: WH-determiner
    WP: WH-pronoun
    WP$: WH-pronoun, possessive

    Other that are kept, because they are not "words",
    but rather misc things we might want to keep:
    $: dollar
    '': closing quotation mark
    ``: opening quotation mark
    (: opening parenthesis
    ): closing parenthesis
    ,: comma
    --: dash
    .: sentence terminator
    :: colon or ellipsis

    References:
    - https://en.wikipedia.org/wiki/Part_of_speech#Functional_classification
    - https://www.nltk.org/book/ch05.html
    """
    WORDS_OPEN_CLASSES = ["JJ", "JJR", "JJS", "RB", "RBR", "RBS", "WRB", "NN", "NNP", "NNPS", "NNS", "VB", "VBD", "VBG",
                          "VBN", "VBP", "VBZ", "UH"]
    WORDS_CLOSED_CLASSES_OR_OTHER_MISC = ["CC", "CD", "DT", "EX", "FW", "IN", "LS", "MD", "PDT", "POS", "PRP", "PRP",
                                          "RP", "SYM", "TO", "WDT", "WP", "WP"]
    OTHER_NOT_WORDS = ["$", "''", "``", "(", ")", ",", "--", ".", ":"]

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        # self.all_tags = set()
        return self._tag_all_filter(x)

    def _tag_all_filter(self, X):
        filtered = [self._tag_filter(x) for x in X]
        return filtered

    def _tag_filter(self, x):
        tagged = nltk.pos_tag(x)

        # for (word, tag) in tagged:
        #     self.all_tags.add(tag)

        WORDS_TO_KEEP = KeepOpenClassesOnly.WORDS_OPEN_CLASSES + KeepOpenClassesOnly.OTHER_NOT_WORDS

        return [word for (word, tag) in tagged if tag in WORDS_TO_KEEP]

    def print_more_info(self):
        """
        Print the definition of all NLTK's tags:
        """
        nltk.download('tagsets')
        nltk.help.upenn_tagset()
        print("WORDS_OPEN_CLASSES: ", KeepOpenClassesOnly.WORDS_OPEN_CLASSES)
        print("WORDS_CLOSED_CLASSES_OR_OTHER_MISC: ", KeepOpenClassesOnly.WORDS_CLOSED_CLASSES_OR_OTHER_MISC)
        print("OTHER_NOT_WORDS: ", KeepOpenClassesOnly.OTHER_NOT_WORDS)
        print("WORDS_TO_KEEP = WORDS_OPEN_CLASSES + OTHER_NOT_WORDS")

if __name__ == "__main__":
    #    word_tokenize(["this", "is"])
    pass
