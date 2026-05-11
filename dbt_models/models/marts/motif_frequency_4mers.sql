{{ config(tags=['individual']) }}

with bases as (
    select 'A' as base
    union all
    select 'C' as base
    union all
    select 'G' as base
    union all
    select 'T' as base
),

motifs as (
    select
        b1.base || b2.base || b3.base || b4.base as motif
    from bases as b1
    cross join bases as b2
    cross join bases as b3
    cross join bases as b4
),

sides as (
    select 'five_prime' as side
    union all
    select 'three_prime' as side
),

totals as (
    select
        side,
        sum(count) as total_count
    from {{ source('quant', 'quant__motif_counts') }}
    group by side
)

select
    samples.sample,
    sides.side,
    motifs.motif,
    cast(coalesce(motif_counts.count, 0) as double)
    / nullif(totals.total_count, 0) as motif_frequency
from {{ source('quant', 'quant__samples') }} as samples
cross join sides
cross join motifs
left join {{ source('quant', 'quant__motif_counts') }} as motif_counts
    on motif_counts.side = sides.side
    and motif_counts.motif = motifs.motif
left join totals
    on totals.side = sides.side
