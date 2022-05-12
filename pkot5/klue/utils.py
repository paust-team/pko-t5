def contains_as_sublist(short_list, long_list):
    for i in range(len(long_list)):
        if short_list == long_list[i:i+len(short_list)]:
            return True
    return False
