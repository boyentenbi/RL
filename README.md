# RL-reimplementations
My goal here was just to have some fun reimplementing DDPG (before A3C) and policy gradient. 

The PG implementation is from-scratch. For the DPPG implementation I originally wrote my own code and later 'forked' Patrick Emami's blog post for its better OO design (http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html). 


Advantage-PG on pendulum swing-up:

![Alt Text](https://media.giphy.com/media/3o6nUPNSWqaIytYVPi/giphy.gif)

Full video: https://youtu.be/pf-ATPFff74

# Stuff I learned

* TRPO is really hard to implement
* DPPG is extremely sensitive to hyper-parameters. I couldn't get any experiments to work without exactly following those prescribed.
* Both algorithms are very sensitive to random seed
* It's sometimes better to not use OO design before you've got a working draft
