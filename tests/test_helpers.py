from hippollm.helpers import *

def test_parse_bullet_points():
    text1 = """Here are some bullet points:
    - one
    - two
    - three
    """
    points = parse_bullet_points(text1)
    assert points == ['one', 'two', 'three']
    points = parse_bullet_points(text1, only_first_bullets=True)
    assert points == ['one', 'two', 'three']
    
    text2 = """Here are some bullet points:
    * one
    * two
    * three
    """
    points = parse_bullet_points(text2)
    assert points == ['one', 'two', 'three']
    points = parse_bullet_points(text2, only_first_bullets=True)
    assert points == ['one', 'two', 'three']
    
    text3 = """one
    - two
    - three
    """
    points = parse_bullet_points(text3)
    assert points == ['one', 'two', 'three']
    points = parse_bullet_points(text3, only_first_bullets=True)
    assert points == ['one', 'two', 'three']
    
    text4 = """1. one
    2. two
    3. three
    """
    points = parse_bullet_points(text4)
    assert points == ['one', 'two', 'three']
    points = parse_bullet_points(text4, only_first_bullets=True)
    assert points == ['one', 'two', 'three']
    
    text5 = """one
    two
    three
    """
    points = parse_bullet_points(text5)
    assert points == ['one', 'two', 'three']
    points = parse_bullet_points(text5, only_first_bullets=True)
    assert points == ['one']
    
    text6 = """There are no facts.
    """
    points = parse_bullet_points(text6)
    assert points == []
    points = parse_bullet_points(text6, only_first_bullets=True)
    assert points == []
    
    text7 = """None"""
    points = parse_bullet_points(text7)
    assert points == []
    points = parse_bullet_points(text7, only_first_bullets=True)
    assert points == []
    