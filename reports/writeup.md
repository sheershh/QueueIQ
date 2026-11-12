# Predicting what happens to projects in the US grid queue

*A write-up of QueueIQ. Every number below comes from code in this repository.*

## The question

Before a new solar farm, wind farm, battery or gas plant can sell power, it has to connect to the grid. To do
that it joins an **interconnection queue**: a line of projects waiting for the grid operator to study what each
connection would need. These queues have become very long. Most projects that join one never get built. They
**withdraw**, often after years of waiting.

I wanted to answer a practical question for someone planning a project: *on the day a project joins the queue,
what are the chances it gets built, what are the chances it drops out, and how long will it wait?*

## The data

Lawrence Berkeley National Laboratory publishes *Queued Up*, a yearly dataset of every queue request in the US,
built together with GridTracker. I used the 2026 edition, which covers queues through the end of 2025.

Not every region records when projects withdraw or start operating, and without those dates you can't measure
waiting times. So I kept the four grid operators with good date coverage (PJM, MISO, CAISO and SPP) and projects
that entered from 2010 on. That left **14,796 projects**. By the end of 2025:

- 9,627 had withdrawn;
- 1,230 were operating;
- 3,939 were still waiting.

That last group matters. If you simply dropped the projects still waiting, or counted them as failures, you
would get the wrong answer about both success rates and waiting times.

## Why a "survival" model

This kind of problem has a name: **survival analysis with competing risks**. Each project is on a clock that
starts when it joins the queue. One of two things ends the clock: it reaches operation, or it withdraws. For
projects still waiting, we know the clock has run *at least* this long, and the model uses that partial
information instead of throwing it away.

The output is two curves per project: the chance it has withdrawn by month 1, 2, 3, and so on, and the chance it
is operating by then. I compared four kinds of model on the same data:

- a simple baseline that averages past projects in the same region and fuel;
- a Cox model;
- random survival forests;
- a gradient-boosted model (LightGBM) that predicts month by month.

## Not cheating with time

The biggest risk in a project like this is accidentally letting the model see the future. Three rules guarded
against it:

1. **Only use what was known on entry day.** Size, fuel, location and grid operator are known when a project
   joins. Study results and later status updates are not, so they are never used.
2. **Count the queue as it looked that day.** One set of features describes how crowded the queue was: how many
   requests were filed nearby in the previous five years, and how many had already withdrawn. Those counts only
   include requests and withdrawals dated *before* the project's own entry date. A test rebuilds the data as
   it looked on each project's entry day (later entries removed, later withdrawals unknown, status scrambled)
   and checks that the project's features come out identical.
3. **Test on later years only.** To test on, say, 2017 entrants, the model trains only on earlier projects, with
   their outcomes cut off at the start of 2017, exactly what someone would have known then.

## Deciding the rules before seeing the results

Before computing any final numbers, I wrote down the whole testing plan and saved it in git with a tag:

- which years to develop on;
- how to score models and pick one;
- which years were held back as the final test.

The selected model was frozen on 2014–2017 entrants. Then 2018–2020 were scored for reporting only, and the
2021–22 entrants were scored **once** as the final test.

This matters because if you look at the test results and then change your choice, the test stops being a test.

## What happened

**Predicting which projects get built works reasonably well where it can be checked.** The model ranks projects
by their chance of reaching operation within 3 years. For 2014–2017 entrants, it put a project that got built
above one that didn't about 72% of the time (AUC 0.723; 0.5 would be a coin flip). Across seven test years it
ranged from 0.69 to 0.89. Most of that comes from two things: project size and technology. Gas plants are built
far more often than batteries.

**For the newest projects, it can't be checked.** Of the 3,299 projects that entered in 2021–22, exactly one was
operating three years later. You can't measure how well a model ranks one success.

**Predicting withdrawal barely beats a coin flip.** On the final test, the withdrawal AUC was 0.559. Across the
earlier test years it ranged from 0.47 to 0.60. My reading is that withdrawals are decided by things this data
doesn't contain: financing, power-purchase contracts, and the grid-upgrade costs that studies reveal later.

**The probabilities run too high.** For 2021–22 entrants, the model said 51.6% would withdraw within 3 years.
The real figure was 34.3%. It said 4.6% would be operating, and the real figure was 0.03%. It over-predicted
withdrawal in every year I tested. I checked whether missing withdrawal dates in the data could explain this.
They account for at most 3% of the gap.

**My pre-registered choice did not hold up.** On the final test, 12 of the 14 other candidates scored better on
withdrawal than the model I had selected. The best was a Cox model that used the entry year and a simple
recalibration step. I did not switch, because switching after seeing the final test would make it a tuning
set. The honest move is to name that model in advance next time and test it on years that haven't happened yet.

**The queue-crowding features didn't help.** I expected that "how crowded the queue was nearby" would predict
withdrawal. For the forest model, the version without those features scored better on withdrawal, in
development and on the final test. They do change individual predictions, which the explanation tab shows, but
that is not the same as improving them.

## The dashboard

The dashboard is live at https://queueiq.streamlit.app. You click a county on a 3D map of the covered
states: column height is how many projects asked to connect there recently, color is how many of those
already gave up. Then you pick the technology and size.

The first version led with the model's percentages, and that was the wrong thing to put first. Those
percentages are exactly where the model is weakest. So the redesigned page leads with **what actually
happened** to comparable past projects, as a sentence and a grid of 100 squares. For example: of 2,163 PJM
solar projects that entered in 2019–2022, 32% had withdrawn within 3 years, under 1% were operating, and 68%
were still waiting.

The model only does what it is somewhat good at: ranking this project against recent entrants. Each ranking
says how far it can be trusted, in plain words: in back-tests, the model put the right project first 72% of
the time for reaching operation (moderate), but only 56% for withdrawal (weak, where a coin flip is 50%).

Three choices keep it honest:

- **The model's curves come with a band built from its own past mistakes.** For each past test year I
  measured how far off its average prediction was, then redrew the curves with that error. The band shows
  where the answer lands if the model is as wrong as it has been before. It is not a confidence interval.
  When I checked it against the 2021–22 projects, it covered the withdrawal rate but not the operation
  rate, and the dashboard says so.
- **The unchecked part is greyed out.** Predictions were only ever tested up to 5 years, so the curves
  beyond that are marked as unchecked.
- **Nothing is assumed silently.** If you don't pick a county, the busiest county in the state is used, and
  the screen names it.

## What I learned

- **A careful negative result is still a result.** The pipeline, the leakage tests and the pre-registered
  evaluation are the parts I'd reuse. The finding that withdrawal is hard to predict from entry-day facts is
  useful in itself.
- **The world shifted under the model.** In these four regions, new requests rose from 631 in 2015 to 1,720 in
  2021, and the rules changed (for example FERC Order 2023). A model trained on the past gets the level wrong
  for the present, even when its ranking holds.
- **Uncertainty should come from evidence.** A band built from real past errors is easier to defend than one
  that only measures how much a model disagrees with itself.

## What's next

- Re-test when LBNL publishes the next edition. It will show 2023 entrants at three years, which is the first
  real check of the deployed model and of the band.
- Write a second pre-registered plan that names the recalibrated Cox model in advance.
- Add information closer to where withdrawals are decided, where it is known at entry, such as substation-level
  crowding from the grid operators' own queue files.

*Data: Queued Up Data File, Copyright (c) 2026, The Regents of the University of California, through Lawrence
Berkeley National Laboratory & GridTracker, licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The data was filtered, cleaned and modelled for this project. LBNL and GridTracker do not endorse it.*
