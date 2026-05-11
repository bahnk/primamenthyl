{{ config(tags=['individual']) }}

with bins as (
    select generate_series as bin
    from generate_series(0, 46700000, 100000)
),

counts as (
    select
        floor(((start_position + end_position) / 2.0) / 100000) * 100000 as bin,
        count(*) as fragment_count
    from {{ ref('stg_query_fragments') }}
    group by 1
),

totals as (
    select count(*) as total_fragment_count
    from {{ ref('stg_query_fragments') }}
)

select
    samples.sample,
    bins.bin,
    cast(coalesce(counts.fragment_count, 0) as double)
    / nullif(totals.total_fragment_count, 0) as fragment_fraction
from {{ source('quant', 'quant__samples') }} as samples
cross join bins
cross join totals
left join counts
    on bins.bin = counts.bin
