SELECT assignment_id, reviewer_UID, grade_for_reviewer 
FROM isdata1.review_grades
where grade_for_reviewer <> 'NULL'
AND grade_for_reviewer IS NOT NULL
order by assignment_id; 

select count(*) from review_comments

SELECT distinct(assignment_id) 
FROM isdata1.review_grades
where grade_for_reviewer <> 'NULL'
AND grade_for_reviewer IS NOT NULL
order by assignment_id; 

SELECT count(*) FROM isdata1.review_comments;

select * from
(select assignment_id,  reviewer_UID, count(*) as Count from review_grades
#where count(*) <> 1
group by assignment_id, reviewer_UID) as temp1
#where temp1. Count <>1

insert into reviews_combined 
(assignment_id, reviewer_uid, reviewee_team, comments,scores, questions)
SELECT assignment_id, reviewer_uid,reviewee_team, GROUP_CONCAT(comment SEPARATOR ' '), GROUP_CONCAT(score SEPARATOR ','), GROUP_CONCAT(question SEPARATOR ' ')  FROM isdata1.review_comments GROUP BY assignment_id, reviewer_uid,reviewee_team;

select * from isdata1.review_grades where assignment_id = 152 and reviewer_UID = 'srbriggs';
select * from isdata1.review_comments where assignment_id = 39 and reviewer_UID = 'jvegafr';

select * from review_grades where id = 1550
delete from review_grades where id = 1550

select distinct(assignment_id) from isdata1.review_comments where assignment_id in (19,36,39,76,152,820,824,827)

#################################################################################
select temp.assignment_id, temp.reviewer_uid, temp.reviewee_team, temp.scores, temp.comments, temp.questions, r.grade_for_reviewer from
(SELECT assignment_id, reviewer_uid,reviewee_team, GROUP_CONCAT(comment SEPARATOR ' ') as comments, GROUP_CONCAT(score SEPARATOR ',') as scores, GROUP_CONCAT(question SEPARATOR ' ') as questions  FROM review_comments GROUP BY assignment_id, reviewer_uid,reviewee_team)
as temp	
left join review_grades r on temp.assignment_id = r.assignment_id AND temp.reviewer_uid = r.reviewer_uid
where r.grade_for_reviewer <> 'NULL' AND r.grade_for_reviewer IS NOT NULL
