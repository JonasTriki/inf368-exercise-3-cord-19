def fix_authors(authors: str, authors_sep: str = ';', name_sep: str = ','):
    authors_split = authors.split(authors_sep)
    if len(authors_split) > 2:

        # Use first authors last name + et al.
        return f'{authors_split[0].split(name_sep)[0]} et al.'
    else:
        
        # Separate authors using comma (,)
        return ', '.join(authors_split)