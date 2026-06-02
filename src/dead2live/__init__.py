"""dead2live - image/text -> controllable 2D digital human.

Pipeline:  instruction text --(Brain)--> AnimationState timeline
           portrait image  --(Rig)----> feature rig + palette
                                          |
                                  (PuppetAnimator) --> frames -> mp4/gif
"""
__version__ = "0.1.0"
