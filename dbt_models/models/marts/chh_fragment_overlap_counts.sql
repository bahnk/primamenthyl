{{ config(tags=['individual']) }}

with methylation_positions as (
    select distinct
        reference,
        position
    from {{ ref('stg_methylation_positions') }}
    where context = 'chh'
),

fragment_references as (
    select distinct
        query_id,
        reference
    from {{ source('quant', 'quant__records') }}
),

fragments as (
    select
        fragment_references.reference,
        query_fragments.query_id,
        query_fragments.start_position,
        query_fragments.end_position
    from {{ ref('stg_query_fragments') }} as query_fragments
    inner join fragment_references
        on query_fragments.query_id = fragment_references.query_id
)

select
    samples.sample,
    methylation_positions.reference,
    methylation_positions.position,
    count(fragments.query_id) as fragment_overlap_count
from {{ source('quant', 'quant__samples') }} as samples
cross join methylation_positions
left join fragments
    on methylation_positions.reference = fragments.reference
    and methylation_positions.position between fragments.start_position and fragments.end_position
group by samples.sample, methylation_positions.reference, methylation_positions.position
