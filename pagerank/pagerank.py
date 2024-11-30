import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Generate a probability distribution for the next page visit based on the current page.

    With probability `damping_factor`, select a link at random from the current page's links.
    With probability `1 - damping_factor`, select a random page from the entire corpus.

    If the current page has no outgoing links, assign equal probability to all pages in the corpus.
    """
    # Initialize a dictionary to hold the probability distribution for each page.
    probabilities = {p: 0 for p in corpus}

    # If the current page has no links, distribute probabilities equally across all corpus pages.
    if not corpus[page]:
        equal_prob = 1 / len(corpus)
        probabilities = {p: equal_prob for p in corpus}
        return probabilities

    # Calculate the probabilities for selecting any page at random and for selecting a link on the page.
    random_choice_prob = (1 - damping_factor) / len(corpus)
    link_choice_prob = damping_factor / len(corpus[page])

    # Assign probabilities to each page based on the above conditions.
    for p in corpus:
        probabilities[p] = random_choice_prob + (link_choice_prob if p in corpus[page] else 0)

    return probabilities



def sample_pagerank(corpus, damping_factor, n):
    """
    Estimate PageRank values for each page by sampling `n` pages
    according to the transition model, starting with a random page.

    Returns a dictionary where keys are page names, and values are
    the estimated PageRank value (between 0 and 1), with all values summing to 1.
    """
    # Initialize visit counts for each page.
    visit_counts = {page: 0 for page in corpus}

    # Start with a randomly selected page.
    current_page = random.choice(list(corpus.keys()))
    visit_counts[current_page] += 1

    # Perform sampling to estimate PageRank.
    for _ in range(n - 1):
        # Get the transition model for the current page.
        transition_probs = transition_model(corpus, current_page, damping_factor)

        # Select the next page based on the transition model probabilities.
        random_choice = random.random()
        cumulative_prob = 0
        for page, prob in transition_probs.items():
            cumulative_prob += prob
            if random_choice <= cumulative_prob:
                current_page = page
                break

        # Update the visit count for the selected page.
        visit_counts[current_page] += 1

    # Convert visit counts to probabilities by dividing by the sample count.
    page_ranks = {page: count / n for page, count in visit_counts.items()}

    print('Sum of sample page ranks:', round(sum(page_ranks.values()), 4))
    return page_ranks


def iterate_pagerank(corpus, damping_factor):
    """
    Calculate PageRank values for each page by iteratively updating
    values until convergence.

    Returns a dictionary where keys are page names, and values are
    the estimated PageRank (between 0 and 1), with values summing to 1.
    """
    num_pages = len(corpus)
    initial_rank = 1 / num_pages
    random_jump_prob = (1 - damping_factor) / num_pages
    page_ranks = {page: initial_rank for page in corpus}

    change_threshold = 0.001
    max_change = change_threshold
    iteration_count = 0

    while max_change >= change_threshold:
        iteration_count += 1
        max_change = 0
        new_ranks = {}

        # Calculate the new rank for each page.
        for page in corpus:
            link_prob_sum = sum(
                page_ranks[linked_page] / len(corpus[linked_page])
                for linked_page in corpus if page in corpus[linked_page]
            )
            # Add probability from pages with no outbound links.
            link_prob_sum += sum(
                page_ranks[no_link_page] / num_pages for no_link_page in corpus if not corpus[no_link_page]
            )

            new_rank = random_jump_prob + damping_factor * link_prob_sum
            new_ranks[page] = new_rank

        # Normalize new ranks and calculate the maximum change.
        norm_factor = sum(new_ranks.values())
        for page in new_ranks:
            new_ranks[page] /= norm_factor
            max_change = max(max_change, abs(page_ranks[page] - new_ranks[page]))

        page_ranks = new_ranks

    print('Iterations to converge:', iteration_count)
    print('Sum of iteration page ranks:', round(sum(page_ranks.values()), 4))
    return page_ranks



if __name__ == "__main__":
    main()
