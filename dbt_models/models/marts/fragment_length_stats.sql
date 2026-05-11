{{ config(tags=['individual']) }}

with fragment_lengths as (
    select
        abs(end_position - start_position) as fragment_length
    from {{ ref('stg_query_fragments') }}
)

select
    samples.sample,
    avg(fragment_lengths.fragment_length) as mean_fragment_length,
    median(fragment_lengths.fragment_length) as median_fragment_length
from {{ source('quant', 'quant__samples') }} as samples
cross join fragment_lengths
group by samples.sample
