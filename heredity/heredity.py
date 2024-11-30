import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return the joint probability that:
        * everyone in `one_gene` has one gene copy,
        * everyone in `two_genes` has two gene copies,
        * everyone else has zero gene copies,
        * everyone in `have_trait` exhibits the trait, and
        * everyone not in `have_trait` does not exhibit the trait.
    """
    joint_prob = 1

    # Calculate probabilities for each person
    for person in people:
        gene_count = 2 if person in two_genes else 1 if person in one_gene else 0
        has_trait = person in have_trait

        # Determine gene probability
        if not people[person]['mother'] and not people[person]['father']:
            # Use unconditional gene probability if no parent information
            prob = PROBS['gene'][gene_count]
        else:
            # Calculate gene inheritance probability based on parents
            mother = people[person]['mother']
            father = people[person]['father']
            mother_prob = inherit_prob(mother, one_gene, two_genes)
            father_prob = inherit_prob(father, one_gene, two_genes)

            # Calculate probability based on gene count
            if gene_count == 2:
                prob = mother_prob * father_prob
            elif gene_count == 1:
                prob = mother_prob * (1 - father_prob) + (1 - mother_prob) * father_prob
            else:
                prob = (1 - mother_prob) * (1 - father_prob)

        # Multiply by the probability of having/not having the trait
        prob *= PROBS['trait'][gene_count][has_trait]

        # Update the joint probability
        joint_prob *= prob

    return joint_prob


def inherit_prob(parent, one_gene, two_genes):
    """
    Returns the probability of a parent passing on a gene copy.
    """
    if parent in two_genes:
        return 1 - PROBS['mutation']
    elif parent in one_gene:
        return 0.5
    else:
        return PROBS['mutation']


def update(probabilities, one_gene, two_genes, have_trait, joint_prob):
    """
    Add the `joint_prob` to `probabilities` for each person's gene and trait counts.
    """
    for person in probabilities:
        gene_count = 2 if person in two_genes else 1 if person in one_gene else 0
        has_trait = person in have_trait

        probabilities[person]['gene'][gene_count] += joint_prob
        probabilities[person]['trait'][has_trait] += joint_prob


def normalize(probabilities):
    """
    Normalize `probabilities` so each probability distribution sums to 1.
    """
    for person in probabilities:
        # Normalize gene distribution
        gene_total = sum(probabilities[person]['gene'].values())
        probabilities[person]['gene'] = {gene: prob / gene_total for gene, prob in probabilities[person]['gene'].items()}

        # Normalize trait distribution
        trait_total = sum(probabilities[person]['trait'].values())
        probabilities[person]['trait'] = {trait: prob / trait_total for trait, prob in probabilities[person]['trait'].items()}


if __name__ == "__main__":
    main()
